"""
auth.py — JWT Authentication & Role-Based Access Control
=========================================================
Features:
  - JWT token authentication
  - Password hashing with bcrypt
  - Role-based access (admin / manager / viewer)
  - User management (add, list, delete users)
  - Token expiry (8 hours)

Install:
  pip install python-jose[cryptography] passlib[bcrypt] python-multipart

Usage:
  from auth import get_current_user, require_role, router as auth_router
  app.include_router(auth_router)
"""

import sqlite3
import secrets
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# Change this secret key in production!
SECRET_KEY    = secrets.token_hex(32)
ALGORITHM     = "HS256"
TOKEN_EXPIRE  = 8  # hours

pwd_context   = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
router        = APIRouter(prefix="/auth", tags=["Authentication"])

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    return conn

def init_users_db():
    """Create users table and default accounts."""
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            username     TEXT UNIQUE NOT NULL,
            email        TEXT UNIQUE NOT NULL,
            full_name    TEXT,
            hashed_pw    TEXT NOT NULL,
            role         TEXT NOT NULL DEFAULT 'viewer',
            is_active    INTEGER NOT NULL DEFAULT 1,
            created_at   TEXT NOT NULL,
            last_login   TEXT
        )
    """)
    conn.commit()

    # Create default users if none exist
    existing = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if existing == 0:
        default_users = [
            ("admin",   "admin@logisense.com",   "Admin User",    "admin123",   "admin"),
            ("manager", "manager@logisense.com", "Manager User",  "manager123", "manager"),
            ("viewer",  "viewer@logisense.com",  "Viewer User",   "viewer123",  "viewer"),
        ]
        for username, email, full_name, password, role in default_users:
            conn.execute("""
                INSERT INTO users (username, email, full_name, hashed_pw, role, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (username, email, full_name, pwd_context.hash(password), role, datetime.now().isoformat()))
        conn.commit()
        print("✅ Default users created:")
        print("   admin   / admin123   (Admin)")
        print("   manager / manager123 (Manager)")
        print("   viewer  / viewer123  (Viewer)")

    conn.close()

# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────

class Token(BaseModel):
    access_token: str
    token_type:   str
    role:         str
    username:     str
    full_name:    str
    expires_in:   int  # seconds

class UserOut(BaseModel):
    id:         int
    username:   str
    email:      str
    full_name:  str
    role:       str
    is_active:  bool
    created_at: str
    last_login: Optional[str]

class UserCreate(BaseModel):
    username:  str
    email:     str
    full_name: str
    password:  str
    role:      str = "viewer"

class PasswordChange(BaseModel):
    current_password: str
    new_password:     str

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def create_token(data: dict, expires_hours: int = TOKEN_EXPIRE) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(hours=expires_hours)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def get_user(username: str) -> Optional[dict]:
    conn = get_db()
    row  = conn.execute("SELECT * FROM users WHERE username=? AND is_active=1", (username,)).fetchone()
    conn.close()
    return dict(row) if row else None

def authenticate_user(username: str, password: str) -> Optional[dict]:
    user = get_user(username)
    if not user or not verify_password(password, user["hashed_pw"]):
        return None
    return user

# ─────────────────────────────────────────────
# CURRENT USER DEPENDENCY
# ─────────────────────────────────────────────

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """Decode JWT and return current user. Raises 401 if invalid."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token. Please log in again.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload  = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = get_user(username)
    if not user:
        raise credentials_exception
    return user

def require_role(*allowed_roles: str):
    """Dependency factory — restricts endpoint to specific roles."""
    async def role_checker(current_user: dict = Depends(get_current_user)) -> dict:
        if current_user["role"] not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {list(allowed_roles)}. Your role: {current_user['role']}"
            )
        return current_user
    return role_checker

# ─────────────────────────────────────────────
# AUTH ENDPOINTS
# ─────────────────────────────────────────────

@router.post("/login", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    """Login with username and password. Returns JWT token."""
    user = authenticate_user(form.username, form.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Update last login
    conn = get_db()
    conn.execute("UPDATE users SET last_login=? WHERE username=?",
                 (datetime.now().isoformat(), user["username"]))
    conn.commit()
    conn.close()

    token = create_token({"sub": user["username"], "role": user["role"]})
    return Token(
        access_token=token,
        token_type="bearer",
        role=user["role"],
        username=user["username"],
        full_name=user["full_name"] or user["username"],
        expires_in=TOKEN_EXPIRE * 3600,
    )

@router.get("/me", response_model=UserOut)
async def get_me(current_user: dict = Depends(get_current_user)):
    """Get current logged-in user details."""
    return UserOut(**{k: current_user[k] for k in UserOut.model_fields if k in current_user})

@router.post("/change-password")
async def change_password(
    data: PasswordChange,
    current_user: dict = Depends(get_current_user)
):
    """Change current user's password."""
    if not verify_password(data.current_password, current_user["hashed_pw"]):
        raise HTTPException(400, "Current password is incorrect.")
    conn = get_db()
    conn.execute("UPDATE users SET hashed_pw=? WHERE username=?",
                 (hash_password(data.new_password), current_user["username"]))
    conn.commit()
    conn.close()
    return {"message": "Password changed successfully."}

# ─────────────────────────────────────────────
# USER MANAGEMENT (Admin only)
# ─────────────────────────────────────────────

@router.get("/users", response_model=list[UserOut])
async def list_users(current_user: dict = Depends(require_role("admin"))):
    """List all users. Admin only."""
    conn  = get_db()
    rows  = conn.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
    conn.close()
    return [UserOut(**{k: dict(r)[k] for k in UserOut.model_fields if k in dict(r)}) for r in rows]

@router.post("/users", response_model=UserOut)
async def create_user(
    user: UserCreate,
    current_user: dict = Depends(require_role("admin"))
):
    """Create a new user. Admin only."""
    if user.role not in ["admin", "manager", "viewer"]:
        raise HTTPException(400, "Role must be: admin, manager, or viewer.")
    try:
        conn = get_db()
        conn.execute("""
            INSERT INTO users (username, email, full_name, hashed_pw, role, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user.username, user.email, user.full_name,
              hash_password(user.password), user.role, datetime.now().isoformat()))
        conn.commit()
        row = conn.execute("SELECT * FROM users WHERE username=?", (user.username,)).fetchone()
        conn.close()
        return UserOut(**{k: dict(row)[k] for k in UserOut.model_fields if k in dict(row)})
    except sqlite3.IntegrityError:
        raise HTTPException(400, "Username or email already exists.")

@router.delete("/users/{username}")
async def delete_user(
    username: str,
    current_user: dict = Depends(require_role("admin"))
):
    """Deactivate a user. Admin only."""
    if username == current_user["username"]:
        raise HTTPException(400, "Cannot deactivate your own account.")
    conn = get_db()
    conn.execute("UPDATE users SET is_active=0 WHERE username=?", (username,))
    conn.commit()
    conn.close()
    return {"message": f"User '{username}' deactivated successfully."}

@router.put("/users/{username}/role")
async def update_role(
    username: str,
    role: str,
    current_user: dict = Depends(require_role("admin"))
):
    """Change a user's role. Admin only."""
    if role not in ["admin", "manager", "viewer"]:
        raise HTTPException(400, "Role must be: admin, manager, or viewer.")
    conn = get_db()
    conn.execute("UPDATE users SET role=? WHERE username=?", (role, username))
    conn.commit()
    conn.close()
    return {"message": f"User '{username}' role updated to '{role}'."}
