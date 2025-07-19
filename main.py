import resend
import os
import io
import asyncio
import threading
import queue
import numpy as np
import time
from fastapi import FastAPI, WebSocket, Depends, HTTPException, Request, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime, Text, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
from pydantic import BaseModel
from jose import JWTError, jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext
from dotenv import load_dotenv
from starlette.websockets import WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse, RedirectResponse
from uuid import uuid4
from google.cloud import speech_v1p1beta1 as speech
from google.oauth2 import service_account
import PyPDF2
from fastapi import Query
import razorpay
import hmac
import hashlib
import aiosmtplib
from email.message import EmailMessage
import secrets
from jose import jwt
import re
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import json

# Load environment variables
load_dotenv()
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "golden-rush-461107-g3-9df6d30285fd.json"
creds_json_str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
creds_info = json.loads(creds_json_str)

# credentials = speech.Credentials.from_service_account_info(json.loads(creds_json))
credentials = service_account.Credentials.from_service_account_info(creds_info)

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

RESET_TOKEN_SECRET = os.getenv("RESET_TOKEN_SECRET", "reset-secret")
RESET_TOKEN_EXPIRE_MINUTES = int(os.getenv("RESET_TOKEN_EXPIRE_MINUTES", 15))
FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL", "http://localhost:3000")

resend.api_key = os.getenv("RESEND_API_KEY")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)



DATABASE_URL = os.getenv("DATABASE_URL")  # e.g., from Render env vars

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Replace with your Razorpay Key ID and Key Secret
RAZORPAY_KEY_ID = 'rzp_test_SFWYv6tj2XbE5u'
RAZORPAY_SECRET_KEY = os.getenv("RAZORPAY_SECRET_KEY")

client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_SECRET_KEY))

# ========================== EMAIL Functions ==========================

import aiosmtplib
from email.message import EmailMessage
from fastapi import BackgroundTasks

from resend import Emails

async def send_email_async(subject: str, recipient: str, content: str, html_content: str = None):
    try:
        payload = {
            "from": "Crackify <support@crackify-ai.com>",
            "to": [recipient],
            "subject": subject,
            "html": html_content or f"<pre>{content}</pre>"
        }
        Emails.send(payload)
    except Exception as e:
        print(f"❌ Failed to send email via Resend: {e}")


# ========================== MODELS ==========================

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    subscription_status = Column(String, default="free")
    live_sessions_remaining = Column(Integer, default=0)
    is_verified = Column(Boolean, default=False)
    verification_token = Column(String, nullable=True)

class InterviewSession(Base):
    __tablename__ = "interview_sessions"
    id = Column(String, primary_key=True, index=True)
    user_email = Column(String, index=True)
    company = Column(String)
    role = Column(String)
    language = Column(String)
    timestamp = Column(String, default=datetime.utcnow)

    questions = relationship("SessionQuestion", back_populates="session")

class SessionQuestion(Base):
    __tablename__ = "session_questions"
    id = Column(String, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("interview_sessions.id"))
    question = Column(String)
    summarized_question = Column(Text)  # NEW
    answer = Column(Text)                 # NEW
    timestamp = Column(DateTime)

    session = relationship("InterviewSession", back_populates="questions")

class Payment(Base):
    __tablename__ = "payments"
    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(String)
    razorpay_order_id = Column(String)
    razorpay_payment_id = Column(String)
    plan = Column(String)
    amount = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ========================== password validation ==========================

def is_valid_password(password: str) -> bool:
    # At least 8 characters, and contains both letters and numbers
    return len(password) >= 8 and re.search("[a-zA-Z]", password) and re.search("[0-9]", password)


# ========================== AUTH ==========================

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_password_hash(password): return pwd_context.hash(password)
def verify_password(plain, hashed): return pwd_context.verify(plain, hashed)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401)
    except JWTError:
        raise HTTPException(status_code=401)
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise HTTPException(status_code=401)
    return user

# ========================== API SCHEMAS ==========================

class UserCreate(BaseModel):
    email: str
    password: str

class QuestionRequest(BaseModel):
    question: str
    session_id: str
# ========================== HTML Templates ==========================

def get_html_email_body(title, message, footer="Thanks, Crackify Team"):
    return f"""
    <html>
      <body style="font-family:Arial,sans-serif; background-color:#f5f5f5; padding:20px;">
        <div style="max-width:600px; margin:auto; background:white; padding:30px; border-radius:8px;">
          <div style="text-align:center;">
            <img src='https://i.postimg.cc/4yT60QSq/logo.png' alt='Crackify Logo' width='120'/>
          </div>
          <h2 style="color:#2563eb; text-align:center;">{title}</h2>
          <p style="font-size:16px; color:#333;">{message}</p>
          <p style="font-size:14px; color:#888;">{footer}</p>
        </div>
      </body>
    </html>
    """


# ========================== AUTH ROUTES ==========================

@app.post("/register")
@limiter.limit("5/minute")
def register(request: Request, user: UserCreate, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    if not is_valid_password(user.password):
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 8 characters long and include both letters and numbers."
        )

    new_user = User(email=user.email, hashed_password=get_password_hash(user.password))
    verification_token = secrets.token_urlsafe(32)
    new_user.verification_token = verification_token
    db.add(new_user)
    db.commit()

    # ✅ Send verification email
    verify_link = f"{FRONTEND_BASE_URL}/verify-email/{verification_token}"
    html_body = get_html_email_body(
        title="Verify your Crackify Email",
        message=f"""
            <p>Thank you for registering with Crackify AI.</p>
            <p>Please verify your email by clicking the link below:</p>
            <p><a href="{verify_link}">Verify Email</a></p>
        """
    )
    background_tasks.add_task(send_email_async, "Verify Your Email", new_user.email, "Please verify your Crackify account", html_body)

    return {"message": "Registered successfully. Please check your email to verify your account."}

@app.get("/verify-email/{token}")
def verify_email(token: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.verification_token == token).first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid or expired token")

    user.is_verified = True
    user.verification_token = None
    db.commit()

    return RedirectResponse(url=f"{FRONTEND_BASE_URL}/email-verified", status_code=302)



@app.post("/login")
@limiter.limit("10/minute")
def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    if not user.is_verified:
        raise HTTPException(status_code=403, detail="Please verify your email before logging in.")
    
    token = create_access_token({"sub": user.email})
    return {"access_token": token, "token_type": "bearer"}


# ========================== SESSION HANDLING ==========================

@app.post("/setup_interview")
async def setup_interview(
    company: str = Form(...),
    role: str = Form(...),
    language: str = Form(...),
    resume: UploadFile = File(None),
    job_description: UploadFile = File(None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    resume_text = ""
    jd_text = ""

    if resume:
        resume_bytes = await resume.read()
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(resume_bytes))
            for page in pdf_reader.pages:
                resume_text += page.extract_text() or ""
        except Exception as e:
            print(f"⚠️ Error reading resume PDF: {e}")

    if job_description:
        jd_bytes = await job_description.read()
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(jd_bytes))
            for page in pdf_reader.pages:
                jd_text += page.extract_text() or ""
        except Exception as e:
            print(f"⚠️ Error reading JD PDF: {e}")

    session_id = str(uuid4())
    new_session = InterviewSession(
        id=session_id,
        user_email=user.email,
        company=company,
        role=role,
        language=language
    )
    db.add(new_session)
    db.commit()

    return {
        "session_id": session_id,
        "company": company,
        "role": role,
        "language": language,
        "resume_text": resume_text,
        "jd_text": jd_text
    }

@app.post("/save_session")
async def save_session(request: Request, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    data = await request.json()
    session_id = data.get("session_id")
    questions = data.get("questions", [])

    for item in questions:
        new_question = SessionQuestion(
            id=str(uuid4()),
            session_id=session_id,
            question=item.get("question", ""),
            summarized_question=item.get("summary", ""),  # ← NEW: summarized question
            answer=item.get("answer", ""),                # ← NEW: GPT answer
            timestamp=datetime.utcnow()
        )
        db.add(new_question)

    db.commit()
    return {"msg": "Session saved"}


@app.get("/sessions")
def get_sessions(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    sessions = db.query(InterviewSession).filter(InterviewSession.user_email == user.email).order_by(InterviewSession.timestamp.desc()).all()
    return [{"id": s.id, "company": s.company, "role": s.role, "timestamp": s.timestamp} for s in sessions]

@app.get("/sessions/{session_id}")
def get_session_questions(session_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    session = db.query(InterviewSession).filter(InterviewSession.id == session_id, InterviewSession.user_email == user.email).first()
    if not session:
        return JSONResponse(status_code=404, content={"message": "Session not found"})

    questions = db.query(SessionQuestion).filter(SessionQuestion.session_id == session_id).order_by(SessionQuestion.timestamp).all()

    return {
        "session": {
            "company": session.company,
            "role": session.role,
            "timestamp": session.timestamp
        },
        "questions": [
            {
                "question": q.summarized_question.strip() if q.summarized_question else "",
                "answer": q.answer.strip() if q.answer else ""
            }
            for q in questions
        ]
    }


@app.get("/download_session/{session_id}")
def download_session(session_id: str, token: str = Query(...), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_email = payload.get("sub")
        if not user_email:
            return JSONResponse(status_code=401, content={"message": "Unauthorized"})
    except JWTError:
        return JSONResponse(status_code=401, content={"message": "Unauthorized"})

    session = db.query(InterviewSession).filter(InterviewSession.id == session_id, InterviewSession.user_email == user_email).first()
    if not session:
        return JSONResponse(status_code=404, content={"message": "Session not found"})

    questions = db.query(SessionQuestion).filter(SessionQuestion.session_id == session_id).order_by(SessionQuestion.timestamp).all()

    content = f"Company: {session.company}\nRole: {session.role}\nDate: {session.timestamp}\n\n"
    for idx, q in enumerate(questions, 1):
        content += f"Q{idx}: {q.summarized_question or q.question}\n"
        content += f"{q.answer or ''}\n\n"

    return StreamingResponse(io.BytesIO(content.encode()), media_type="text/plain",
                              headers={"Content-Disposition": f"attachment; filename=session_{session_id}.txt"})



@app.delete("/delete_session/{session_id}")
def delete_session(session_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    session = db.query(InterviewSession).filter(InterviewSession.id == session_id, InterviewSession.user_email == user.email).first()
    if not session:
        return JSONResponse(status_code=404, content={"message": "Session not found"})

    # Delete all questions associated with the session
    db.query(SessionQuestion).filter(SessionQuestion.session_id == session_id).delete()
    db.delete(session)
    db.commit()

    return {"msg": "Session deleted successfully"}

# ========================== RESET PASSWORD ==========================

@app.post("/send_reset_link")
def send_reset_link(request: Request, background_tasks: BackgroundTasks, db: Session = Depends(get_db) ):
    data = asyncio.run(request.json())
    email = data.get("email")

    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="No account found with this email.")

    reset_token = jwt.encode(
        {"sub": email, "exp": datetime.utcnow() + timedelta(minutes=RESET_TOKEN_EXPIRE_MINUTES)},
        RESET_TOKEN_SECRET,
        algorithm=ALGORITHM
    )

    reset_link = f"{FRONTEND_BASE_URL}/reset-password?token={reset_token}"

    html_body = get_html_email_body(
        title="Reset Your Crackify Password",
        message=f"Click the button below to reset your password. This link is valid for {RESET_TOKEN_EXPIRE_MINUTES} minutes.<br><br><a href='{reset_link}' style='background:#2563eb;color:white;padding:10px 20px;border-radius:5px;text-decoration:none;'>Reset Password</a>"
    )
    background_tasks.add_task(
        send_email_async,
        "Crackify Password Reset",
        email,
        f"Reset your password: {reset_link}",
        html_body
    )

    return {"message": "Reset link sent to your email."}

@app.post("/reset_password")
def reset_password(request: Request, db: Session = Depends(get_db)):
    data = asyncio.run(request.json())
    token = data.get("token")
    new_password = data.get("new_password")

    if not token or not new_password:
        raise HTTPException(status_code=400, detail="Token and new password are required.")
    
    if not is_valid_password(new_password):
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 8 characters long and include both letters and numbers."
        )

    try:
        payload = jwt.decode(token, RESET_TOKEN_SECRET, algorithms=[ALGORITHM])
        email = payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=400, detail="Invalid or expired token.")

    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    user.hashed_password = get_password_hash(new_password)
    db.commit()

    return {"message": "Password reset successful. Please log in."}


# ========================== GPT ANSWER ==========================

@app.post("/summarize_question")
async def summarize_question(req: QuestionRequest, user: User = Depends(get_current_user)):
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")

    prompt = f"Please summarize the following interview question into one clear, concise sentence:\n\n{req.question}"

    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        summarized_question = response.choices[0].message.content.strip()
        return {"summarized_question": summarized_question}
    except Exception as e:
        print(f"❌ Summarization Error: {e}")
        return {"summarized_question": req.question}  # fallback to original


@app.post("/generate_answer")
async def generate_answer(req: QuestionRequest, user: User = Depends(get_current_user)):
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")

    async def gpt_streamer():
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4o",
                messages=[{"role": "user", "content": req.question}],
                stream=True,
            )
            async for chunk in response:
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    delta = chunk['choices'][0]['delta']
                    if 'content' in delta:
                        yield delta['content']
        except Exception as e:
            print(f"⚠️ GPT Streaming Error: {e}")
            yield "[Error streaming response]"

    return StreamingResponse(gpt_streamer(), media_type="text/plain")


@app.post('/classify_question')
async def classify_question(request: Request):
    data = await request.json()
    question = data.get('question')

    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")

    prompt = f"""
You are classifying interview questions into one of the following categories:

- "Coding": if the question clearly asks for code, algorithms, data structures, or logic-based solutions.
- "Conceptual" — if it asks you to explain a concept, theory, comparison, or definition (e.g., explain polymorphism, where vs group by).
- "Scenario": if the question asks how you'd handle real-world problems, architecture decisions, infrastructure, DevOps, or design choices.
- "HR": if the question relates to behavior, teamwork, soft skills, strengths/weaknesses, or work ethic.
- "Introduction": if the question is about personal background, such as "Tell me about yourself", career history, or resume walk-through.

Respond with only the category name: Coding, Scenario, HR, Conceptual or Introduction.

Question: {question}
"""

    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        classification = response.choices[0].message.content.strip()
        return {"type": classification}
    except Exception as e:
        print(f"❌ Classification Error: {e}")
        return {"type": "Scenario"}  # fallback


# ========================== subscription ==========================

@app.get("/subscription_status")
def get_subscription_status(user: User = Depends(get_current_user)):
    return {
        "subscription_status": user.subscription_status,
        "live_sessions_remaining": user.live_sessions_remaining
    }

@app.post("/use_live_session")
def use_live_session(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if user.live_sessions_remaining > 0:
        user.live_sessions_remaining -= 1
        db.commit()
        return {"message": "Live session started", "remaining_sessions": user.live_sessions_remaining}
    else:
        raise HTTPException(status_code=403, detail="No live interview sessions remaining. Please subscribe.")


@app.post("/activate_subscription")
async def activate_subscription(
    plan: str = Query('basic'),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    user.subscription_status = plan
    if plan == 'basic':
        user.live_sessions_remaining = 1
    elif plan == 'pro':
        user.live_sessions_remaining = 3
    db.commit()
    return {"message": "Subscription activated successfully"}



@app.post('/create_order')
@limiter.limit("10/minute")
async def create_order(request: Request):
    data = await request.json()
    amount = data.get('amount', 100) * 100  # Razorpay expects amount in paise

    order = client.order.create({
        'amount': amount,
        'currency': 'INR',
        'payment_capture': '1'
    })

    return JSONResponse({'order_id': order['id'], 'razorpay_key': RAZORPAY_KEY_ID})

@app.post("/verify_payment")
@limiter.limit("10/minute")
async def verify_payment(request: Request, background_tasks: BackgroundTasks, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    data = await request.json()
    razorpay_order_id = data.get("razorpay_order_id")
    razorpay_payment_id = data.get("razorpay_payment_id")
    razorpay_signature = data.get("razorpay_signature")
    selected_plan = data.get("selected_plan", "basic")

    secret = RAZORPAY_SECRET_KEY

    generated_signature = hmac.new(
        bytes(secret, 'utf-8'),
        bytes(razorpay_order_id + '|' + razorpay_payment_id, 'utf-8'),
        hashlib.sha256
    ).hexdigest()

    if generated_signature == razorpay_signature:

        base_price = 299 if selected_plan == 'basic' else 599 if selected_plan == 'pro' else 0
        gst_percent = 18
        gst_amount = round(base_price * gst_percent / 100, 2)
        total_amount = base_price + gst_amount

        user.subscription_status = selected_plan
        user.live_sessions_remaining = 3 if selected_plan == 'basic' else 8 if selected_plan == 'pro' else 0

        new_payment = Payment(
            user_email=user.email,
            razorpay_order_id=razorpay_order_id,
            razorpay_payment_id=razorpay_payment_id,
            plan=selected_plan,
            amount=total_amount
        )
        db.add(new_payment)
        db.commit()

        # ✅ Create receipt email HTML
        html_body = get_html_email_body(
            title="Your Crackify AI Subscription is Confirmed!",
            message=f"""
                <p>Thank you for subscribing to the <strong>{selected_plan.title()} Plan</strong>.</p>
                <table style="width:100%; border-collapse: collapse; margin: 20px 0;">
                  <tr>
                    <td><strong>Order ID:</strong></td><td>{razorpay_order_id}</td>
                  </tr>
                  <tr>
                    <td><strong>Payment ID:</strong></td><td>{razorpay_payment_id}</td>
                  </tr>
                  <tr>
                    <td><strong>Date:</strong></td><td>{datetime.utcnow().strftime('%b %d, %Y')}</td>
                  </tr>
                </table>

                <table style="width:100%; border: 1px solid #ccc; padding: 10px; border-collapse: collapse;">
                  <thead>
                    <tr style="background-color: #f5f5f5;">
                      <th style="padding: 8px; border: 1px solid #ccc;">Item</th>
                      <th style="padding: 8px; border: 1px solid #ccc;">Amount (₹)</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td style="padding: 8px; border: 1px solid #ccc;">{selected_plan.title()} Plan</td>
                      <td style="padding: 8px; border: 1px solid #ccc;">{base_price:.2f}</td>
                    </tr>
                    <tr>
                      <td style="padding: 8px; border: 1px solid #ccc;">GST ({gst_percent}%)</td>
                      <td style="padding: 8px; border: 1px solid #ccc;">{gst_amount:.2f}</td>
                    </tr>
                    <tr>
                      <td style="padding: 8px; border: 1px solid #ccc;"><strong>Total</strong></td>
                      <td style="padding: 8px; border: 1px solid #ccc;"><strong>₹{total_amount:.2f}</strong></td>
                    </tr>
                  </tbody>
                </table>

                <p>You can now access all premium features of your plan.</p>
                <p>Need help? Contact us at <a href="mailto:support@crackify-ai.com">support@crackify-ai.com</a>.</p>
            """
        )

        # ✅ Send email receipt
        background_tasks.add_task(
            send_email_async,
            "Crackify AI Subscription Confirmation",
            user.email,
            f"Your {selected_plan} plan is now active. Order ID: {razorpay_order_id}.Total ₹{total_amount:.2f}",
            html_body
        )


        return {"message": "Payment verified and subscription activated successfully."}
    else:
        return JSONResponse(content={"error": "Payment verification failed."}, status_code=400)


@app.get("/payment_history")
def payment_history(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    payments = db.query(Payment).filter(Payment.user_email == user.email).order_by(Payment.timestamp.desc()).all()
    return [{
        "order_id": p.razorpay_order_id,
        "payment_id": p.razorpay_payment_id,
        "plan": p.plan,
        "amount": p.amount,
        "timestamp": p.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    } for p in payments]

# ========================== google_login ==========================

@app.post("/login_google")
def google_login(data: dict, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    email = data.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email required")

    user = db.query(User).filter(User.email == email).first()
    is_new_user = False

    if not user:
        user = User(email=email, hashed_password="GOOGLE_AUTH", is_verified=True)
        db.add(user)
        db.commit()
        is_new_user = True

    if is_new_user:
        html_body = get_html_email_body(
            title="Welcome to Crackify!",
            message="Thank you for registering with Crackify AI via Google Sign-In. We're excited to help you master your interviews!"
        )
        background_tasks.add_task(
            send_email_async,
            "Welcome to Crackify!",
            email,
            "Thank you for registering!",  # plain text fallback
            html_body
        )

    access_token = create_access_token(data={"sub": email})
    return {"access_token": access_token}




# ========================== Email notifications ==========================

# @app.post("/register")
# def register(user: UserCreate, db: Session = Depends(get_db)):
#     if db.query(User).filter(User.email == user.email).first():
#         raise HTTPException(status_code=400, detail="Email already registered")

#     new_user = User(email=user.email, hashed_password=get_password_hash(user.password))
#     db.add(new_user)
#     db.commit()

#     # Send email synchronously
#     try:
#         send_email_sync(user.email)
#     except Exception as e:
#         print(f"❌ Email sending failed: {e}")

#     return {"msg": "Registered"}



# ========================== AUDIO STREAM ==========================

@app.websocket("/audio_stream")
async def audio_stream(websocket: WebSocket):
    await websocket.accept()
    print("🔌 WebSocket connection accepted")

    client = speech.SpeechClient(credentials=credentials)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_automatic_punctuation=True
    )
    streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)

    audio_queue = queue.Queue()
    silence_threshold = 0.002
    silence_duration_sec = 7.5
    last_audio_time = time.time()
    accumulated_chunk = bytearray()
    final_segments = []
    last_partial = ""
    silence_event = threading.Event()

    def request_generator():
        while True:
            chunk = audio_queue.get()
            if chunk is None:
                break
            yield speech.StreamingRecognizeRequest(audio_content=chunk)

    def google_transcriber():
        nonlocal final_segments, last_partial
        try:
            responses = client.streaming_recognize(streaming_config, request_generator())
            for response in responses:
                for result in response.results:
                    if result.is_final:
                        final_text = result.alternatives[0].transcript.strip()
                        if final_text:
                            final_segments.append(final_text)
                            print("✅ Finalized Segment:", final_text)
                    else:
                        partial_text = result.alternatives[0].transcript.strip()
                        if partial_text and partial_text != last_partial:
                            last_partial = partial_text
                            try:
                                if websocket.client_state.name == "CONNECTED":
                                    asyncio.run(websocket.send_json({"type": "partial_transcript", "text": partial_text}))
                            except Exception as e:
                                print(f"❌ WebSocket Send Error (partial): {e}")
                            print("🔹 Partial Transcript:", partial_text)

        except Exception as e:
            print(f"⚠️ Streaming error: {e}")

    def silence_timer():
        nonlocal final_segments, last_partial
        while not silence_event.is_set():
            if time.time() - last_audio_time > silence_duration_sec:
                if final_segments or last_partial:
                    final_question = " ".join(final_segments)
                    if last_partial and (not final_segments or not last_partial.startswith(final_segments[-1])):
                        final_question += " " + last_partial
                    final_question = final_question.strip()
                    print("⏸️ Silence detected. Final Question:", final_question)
                    try:
                        if websocket.client_state.name == "CONNECTED":
                            asyncio.run(websocket.send_json({"type": "final_transcript", "text": final_question}))
                    except Exception as e:
                        print(f"❌ WebSocket Send Error (final): {e}")
                    final_segments = []
                    last_partial = ""
                time.sleep(0.5)
            time.sleep(0.2)

    transcriber_thread = threading.Thread(target=google_transcriber)
    silence_thread = threading.Thread(target=silence_timer)
    transcriber_thread.start()
    silence_thread.start()

    try:
        while True:
            try:
                message = await websocket.receive_bytes()
                audio_queue.put(message)
                accumulated_chunk.extend(message)

                if len(accumulated_chunk) >= int(16000 * 2 * 0.5):
                    audio_np = np.frombuffer(accumulated_chunk, dtype=np.int16).astype(np.float32) / 32768.0
                    rms = np.sqrt(np.mean(audio_np ** 2))
                    if rms > silence_threshold:
                        last_audio_time = time.time()
                    accumulated_chunk.clear()

            except WebSocketDisconnect:
                print("🔌 Client disconnected gracefully.")
                audio_queue.put(None)
                break
            except Exception as e:
                print(f"⚠️ Unexpected streaming error: {e}")
                audio_queue.put(None)
                break

    finally:
        silence_event.set()
        print("INFO: WebSocket connection closed.")

