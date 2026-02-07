#!/usr/bin/env python3
"""
Food Delivery Telegram Bot
Production-ready system with WebApp
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import List, Optional
from datetime import datetime, timezone
import json
import secrets
import string
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import (
    Message, CallbackQuery, WebAppInfo, 
    ReplyKeyboardMarkup, KeyboardButton,
    InlineKeyboardMarkup, InlineKeyboardButton,
    ReplyKeyboardRemove
)
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.redis import RedisStorage
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
import redis.asyncio as redis

from sqlalchemy.ext.asyncio import (
    AsyncSession, create_async_engine, async_sessionmaker
)
from sqlalchemy import select, update, func, and_, or_
from sqlalchemy.orm import selectinload

# Models
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, BigInteger, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship

# ============================================================================
# Configuration
# ============================================================================

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    BOT_TOKEN = os.getenv("BOT_TOKEN",)
    ADMIN_IDS = [int(x.strip()) for x in os.getenv("ADMIN_IDS", "").split(",") if x.strip()]
    DB_URL = os.getenv("DB_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/food_delivery")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    SHOP_CHANNEL_ID = os.getenv("SHOP_CHANNEL_ID")
    COURIER_CHANNEL_ID = os.getenv("COURIER_CHANNEL_ID")
    WEBAPP_URL = os.getenv("WEBAPP_URL", "https://yourdomain.com/webapp")
    MIN_ORDER_AMOUNT = 50000  # 50,000 сум
    
    @staticmethod
    def get_bot_username():
        # This will be set after bot initialization
        return getattr(Config, '_BOT_USERNAME', 'your_bot_username')

config = Config()

# ============================================================================
# Database Setup
# ============================================================================

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    tg_id = Column(BigInteger, unique=True, nullable=False, index=True)
    username = Column(String(100), nullable=True)
    full_name = Column(String(200), nullable=False)
    joined_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    ref_by_user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    
    orders = relationship("Order", back_populates="user")
    referrals = relationship("User", foreign_keys=[ref_by_user_id])
    ref_by = relationship("User", foreign_keys=[ref_by_user_id], remote_side=[id], back_populates="referrals")

class Category(Base):
    __tablename__ = 'categories'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    foods = relationship("Food", back_populates="category")

class Food(Base):
    __tablename__ = 'foods'
    
    id = Column(Integer, primary_key=True)
    category_id = Column(Integer, ForeignKey('categories.id'), nullable=False)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    price = Column(Integer, nullable=False)  # in sum
    rating = Column(Float, default=4.5)
    is_new = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    image_url = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    category = relationship("Category", back_populates="foods")
    order_items = relationship("OrderItem", back_populates="food")

class Order(Base):
    __tablename__ = 'orders'
    
    id = Column(Integer, primary_key=True)
    order_number = Column(String(50), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    customer_name = Column(String(200), nullable=False)
    phone = Column(String(50), nullable=False)
    comment = Column(Text, nullable=True)
    total = Column(Integer, nullable=False)
    status = Column(String(50), nullable=False, default='NEW', index=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    delivered_at = Column(DateTime(timezone=True), nullable=True)
    location_lat = Column(Float, nullable=False)
    location_lng = Column(Float, nullable=False)
    courier_id = Column(Integer, ForeignKey('couriers.id'), nullable=True)
    promo_id = Column(Integer, ForeignKey('promos.id'), nullable=True)
    
    user = relationship("User", back_populates="orders")
    courier = relationship("Courier", back_populates="orders")
    promo = relationship("Promo")
    items = relationship("OrderItem", back_populates="order")

class OrderItem(Base):
    __tablename__ = 'order_items'
    
    id = Column(Integer, primary_key=True)
    order_id = Column(Integer, ForeignKey('orders.id'), nullable=False)
    food_id = Column(Integer, ForeignKey('foods.id'), nullable=True)
    name_snapshot = Column(String(200), nullable=False)
    price_snapshot = Column(Integer, nullable=False)
    qty = Column(Integer, nullable=False, default=1)
    line_total = Column(Integer, nullable=False)
    
    order = relationship("Order", back_populates="items")
    food = relationship("Food", back_populates="order_items")

class Promo(Base):
    __tablename__ = 'promos'
    
    id = Column(Integer, primary_key=True)
    code = Column(String(50), unique=True, nullable=False, index=True)
    discount_percent = Column(Integer, nullable=False)  # 1-90
    expires_at = Column(DateTime(timezone=True), nullable=True)
    usage_limit = Column(Integer, nullable=True)
    used_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    orders = relationship("Order", back_populates="promo")

class Courier(Base):
    __tablename__ = 'couriers'
    
    id = Column(Integer, primary_key=True)
    chat_id = Column(BigInteger, unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    orders = relationship("Order", back_populates="courier")

# ============================================================================
# Database Engine & Session
# ============================================================================

engine = create_async_engine(config.DB_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# ============================================================================
# Redis Storage for FSM
# ============================================================================

redis_client = redis.from_url(config.REDIS_URL)
storage = RedisStorage(redis=redis_client)

# ============================================================================
# Bot & Dispatcher
# ============================================================================

bot = Bot(token=config.BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher(storage=storage)

# Set bot username
async def set_bot_username():
    me = await bot.get_me()
    config._BOT_USERNAME = me.username

# ============================================================================
# FastAPI App
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    await set_bot_username()
    
    # Start polling in background
    asyncio.create_task(dp.start_polling(bot))
    
    yield
    
    # Shutdown
    await bot.session.close()
    await redis_client.close()
    await engine.dispose()

app = FastAPI(lifespan=lifespan, title="Food Delivery API")

# Mount static files for WebApp
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Telegram InitData Verification
# ============================================================================

import hmac
import hashlib
import urllib.parse

def verify_telegram_init_data(init_data: str) -> bool:
    """
    Verify Telegram WebApp initData signature
    """
    try:
        # Parse init data
        parsed = urllib.parse.parse_qs(init_data)
        
        # Get hash from init data
        hash_str = parsed.get('hash', [''])[0]
        if not hash_str:
            return False
        
        # Remove hash from data
        data_pairs = []
        for key, values in parsed.items():
            if key != 'hash':
                data_pairs.append(f"{key}={values[0]}")
        
        # Sort alphabetically
        data_pairs.sort()
        data_check_string = "\n".join(data_pairs)
        
        # Calculate secret
        secret_key = hmac.new(
            b"WebAppData",
            msg=config.BOT_TOKEN.encode(),
            digestmod=hashlib.sha256
        ).digest()
        
        # Calculate hash
        calculated_hash = hmac.new(
            secret_key,
            msg=data_check_string.encode(),
            digestmod=hashlib.sha256
        ).hexdigest()
        
        return calculated_hash == hash_str
    except Exception:
        return False

# ============================================================================
# FastAPI Endpoints for WebApp
# ============================================================================

@app.get("/api/foods")
async def get_foods(init_data: str, category_id: Optional[int] = None, db: AsyncSession = Depends(get_db)):
    """Get foods for WebApp"""
    if not verify_telegram_init_data(init_data):
        raise HTTPException(status_code=401, detail="Invalid initData")
    
    query = select(Food).join(Category).where(
        Food.is_active == True,
        Category.is_active == True
    )
    
    if category_id:
        query = query.where(Food.category_id == category_id)
    
    result = await db.execute(query)
    foods = result.scalars().all()
    
    return [
        {
            "id": food.id,
            "name": food.name,
            "description": food.description,
            "price": food.price,
            "rating": food.rating,
            "is_new": food.is_new,
            "image_url": food.image_url,
            "category_id": food.category_id,
            "category_name": food.category.name
        }
        for food in foods
    ]

@app.get("/api/categories")
async def get_categories(init_data: str, db: AsyncSession = Depends(get_db)):
    """Get categories for WebApp"""
    if not verify_telegram_init_data(init_data):
        raise HTTPException(status_code=401, detail="Invalid initData")
    
    result = await db.execute(
        select(Category).where(Category.is_active == True).order_by(Category.name)
    )
    categories = result.scalars().all()
    
    return [
        {
            "id": cat.id,
            "name": cat.name
        }
        for cat in categories
    ]

@app.post("/api/promo/validate")
async def validate_promo(init_data: str, code: str, db: AsyncSession = Depends(get_db)):
    """Validate promo code"""
    if not verify_telegram_init_data(init_data):
        raise HTTPException(status_code=401, detail="Invalid initData")
    
    result = await db.execute(
        select(Promo).where(
            Promo.code == code,
            Promo.is_active == True,
            or_(
                Promo.expires_at.is_(None),
                Promo.expires_at > datetime.now(timezone.utc)
            )
        )
    )
    promo = result.scalar_one_or_none()
    
    if not promo:
        return {"valid": False, "message": "Promo code not found or expired"}
    
    if promo.usage_limit and promo.used_count >= promo.usage_limit:
        return {"valid": False, "message": "Promo code usage limit reached"}
    
    return {
        "valid": True,
        "discount_percent": promo.discount_percent,
        "code": promo.code
    }

# ============================================================================
# WebApp HTML
# ============================================================================

@app.get("/webapp", response_class=HTMLResponse)
async def webapp_page():
    """Serve WebApp HTML"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FIESTA Food Delivery</title>
    <script src="https://telegram.org/js/telegram-web-app.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --bg-primary: #1a1a1a;
            --bg-secondary: #2d2d2d;
            --bg-tertiary: #3d3d3d;
            --text-primary: #ffffff;
            --text-secondary: #b0b0b0;
            --accent: #ff6b35;
            --accent-hover: #ff8b35;
            --success: #4caf50;
            --danger: #f44336;
            --border: #444444;
            --radius: 12px;
            --shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding-bottom: 100px;
            min-height: 100vh;
        }

        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 16px;
        }

        /* Header */
        .header {
            background: var(--bg-secondary);
            padding: 16px;
            border-radius: var(--radius);
            margin-bottom: 20px;
            box-shadow: var(--shadow);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .search-bar {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }

        .search-input {
            flex: 1;
            padding: 12px 16px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            color: var(--text-primary);
            font-size: 16px;
        }

        .search-input::placeholder {
            color: var(--text-secondary);
        }

        .filter-btn {
            padding: 12px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            color: var(--text-primary);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        /* Categories */
        .categories {
            display: flex;
            gap: 10px;
            overflow-x: auto;
            padding: 10px 0;
            margin-bottom: 20px;
            scrollbar-width: none;
        }

        .categories::-webkit-scrollbar {
            display: none;
        }

        .category-btn {
            padding: 8px 16px;
            background: var(--bg-tertiary);
            border: none;
            border-radius: 20px;
            color: var(--text-primary);
            font-size: 14px;
            cursor: pointer;
            white-space: nowrap;
            transition: all 0.3s ease;
        }

        .category-btn.active {
            background: var(--accent);
            color: white;
        }

        /* Food Items */
        .food-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 100px;
        }

        .food-card {
            background: var(--bg-secondary);
            border-radius: var(--radius);
            overflow: hidden;
            box-shadow: var(--shadow);
            transition: transform 0.3s ease;
        }

        .food-card:hover {
            transform: translateY(-4px);
        }

        .food-image {
            width: 100%;
            height: 180px;
            object-fit: cover;
            background: var(--bg-tertiary);
        }

        .food-info {
            padding: 16px;
        }

        .food-header {
            display: flex;
            justify-content: space-between;
            align-items: start;
            margin-bottom: 8px;
        }

        .food-name {
            font-size: 18px;
            font-weight: 600;
            color: var(--text-primary);
        }

        .food-rating {
            display: flex;
            align-items: center;
            gap: 4px;
            font-size: 14px;
            color: var(--text-secondary);
        }

        .food-desc {
            font-size: 14px;
            color: var(--text-secondary);
            margin-bottom: 12px;
            line-height: 1.4;
        }

        .food-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .food-price {
            font-size: 20px;
            font-weight: 700;
            color: var(--accent);
        }

        .add-btn {
            padding: 8px 20px;
            background: var(--accent);
            color: white;
            border: none;
            border-radius: var(--radius);
            font-weight: 600;
            cursor: pointer;
            transition: background 0.3s ease;
        }

        .add-btn:hover {
            background: var(--accent-hover);
        }

        /* Cart */
        .cart-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: var(--bg-secondary);
            border-top: 1px solid var(--border);
            padding: 16px;
            transform: translateY(100%);
            transition: transform 0.3s ease;
            z-index: 1000;
        }

        .cart-container.active {
            transform: translateY(0);
        }

        .cart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }

        .cart-title {
            font-size: 18px;
            font-weight: 600;
        }

        .close-cart {
            background: none;
            border: none;
            color: var(--text-secondary);
            font-size: 24px;
            cursor: pointer;
            padding: 0;
            width: 30px;
            height: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .cart-items {
            max-height: 300px;
            overflow-y: auto;
            margin-bottom: 16px;
        }

        .cart-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid var(--border);
        }

        .cart-item-info {
            flex: 1;
        }

        .cart-item-name {
            font-weight: 500;
            margin-bottom: 4px;
        }

        .cart-item-price {
            color: var(--text-secondary);
            font-size: 14px;
        }

        .cart-item-controls {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .cart-item-qty {
            font-weight: 600;
            min-width: 30px;
            text-align: center;
        }

        .qty-btn {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            background: var(--bg-tertiary);
            border: none;
            color: var(--text-primary);
            font-size: 18px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .cart-total {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            padding-top: 16px;
            border-top: 1px solid var(--border);
        }

        .total-label {
            font-size: 18px;
            font-weight: 600;
        }

        .total-amount {
            font-size: 24px;
            font-weight: 700;
            color: var(--accent);
        }

        .checkout-btn {
            width: 100%;
            padding: 16px;
            background: var(--accent);
            color: white;
            border: none;
            border-radius: var(--radius);
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.3s ease;
        }

        .checkout-btn:disabled {
            background: var(--bg-tertiary);
            color: var(--text-secondary);
            cursor: not-allowed;
        }

        .checkout-btn:not(:disabled):hover {
            background: var(--accent-hover);
        }

        /* Floating Cart Button */
        .floating-cart {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 60px;
            height: 60px;
            background: var(--accent);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 24px;
            cursor: pointer;
            box-shadow: 0 4px 20px rgba(255, 107, 53, 0.3);
            z-index: 999;
            transition: transform 0.3s ease;
        }

        .floating-cart:hover {
            transform: scale(1.1);
        }

        .cart-count {
            position: absolute;
            top: -5px;
            right: -5px;
            background: var(--danger);
            color: white;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            font-size: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
        }

        /* Checkout Modal */
        .modal {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 2000;
            padding: 20px;
        }

        .modal.active {
            display: flex;
        }

        .modal-content {
            background: var(--bg-secondary);
            border-radius: var(--radius);
            padding: 24px;
            max-width: 500px;
            width: 100%;
            max-height: 90vh;
            overflow-y: auto;
        }

        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }

        .modal-title {
            font-size: 20px;
            font-weight: 600;
        }

        .close-modal {
            background: none;
            border: none;
            color: var(--text-secondary);
            font-size: 24px;
            cursor: pointer;
        }

        .form-group {
            margin-bottom: 16px;
        }

        .form-label {
            display: block;
            margin-bottom: 8px;
            color: var(--text-secondary);
            font-size: 14px;
        }

        .form-input {
            width: 100%;
            padding: 12px 16px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            color: var(--text-primary);
            font-size: 16px;
        }

        .form-input:focus {
            outline: none;
            border-color: var(--accent);
        }

        .location-btn {
            width: 100%;
            padding: 12px 16px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            color: var(--text-primary);
            font-size: 16px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            margin-bottom: 8px;
        }

        .location-btn:hover {
            border-color: var(--accent);
        }

        .location-coords {
            font-size: 12px;
            color: var(--text-secondary);
            margin-bottom: 16px;
            padding: 8px;
            background: var(--bg-tertiary);
            border-radius: 6px;
            display: none;
        }

        .location-coords.active {
            display: block;
        }

        .promo-section {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }

        .promo-input {
            flex: 1;
            padding: 12px 16px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            color: var(--text-primary);
            font-size: 16px;
        }

        .promo-btn {
            padding: 12px 20px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            color: var(--text-primary);
            cursor: pointer;
        }

        .promo-success {
            color: var(--success);
            font-size: 14px;
            margin-bottom: 16px;
            display: none;
        }

        .promo-error {
            color: var(--danger);
            font-size: 14px;
            margin-bottom: 16px;
            display: none;
        }

        .submit-order {
            width: 100%;
            padding: 16px;
            background: var(--accent);
            color: white;
            border: none;
            border-radius: var(--radius);
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.3s ease;
        }

        .submit-order:disabled {
            background: var(--bg-tertiary);
            color: var(--text-secondary);
            cursor: not-allowed;
        }

        .submit-order:not(:disabled):hover {
            background: var(--accent-hover);
        }

        /* Filter Modal */
        .filter-modal {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 2000;
            padding: 20px;
        }

        .filter-modal.active {
            display: flex;
        }

        .filter-options {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .filter-option {
            padding: 16px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            color: var(--text-primary);
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .filter-option:hover {
            border-color: var(--accent);
            background: var(--bg-secondary);
        }

        .filter-option.active {
            background: var(--accent);
            color: white;
            border-color: var(--accent);
        }

        /* Loading */
        .loading {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            align-items: center;
            justify-content: center;
            z-index: 3000;
        }

        .loading.active {
            display: flex;
        }

        .spinner {
            width: 50px;
            height: 50px;
            border: 5px solid var(--bg-tertiary);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Responsive */
        @media (max-width: 768px) {
            .food-grid {
                grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            }
            
            .container {
                padding: 12px;
            }
        }

        @media (max-width: 480px) {
            .food-grid {
                grid-template-columns: 1fr;
            }
            
            .categories {
                gap: 8px;
            }
            
            .category-btn {
                padding: 6px 12px;
                font-size: 13px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 style="margin-bottom: 16px; color: var(--accent);">🍔 FIESTA Delivery</h1>
            
            <div class="search-bar">
                <input type="text" 
                       class="search-input" 
                       placeholder="Search food..."
                       id="searchInput">
                <button class="filter-btn" id="filterBtn">
                    ⚙️
                </button>
            </div>
        </div>

        <!-- Categories -->
        <div class="categories" id="categoriesContainer">
            <!-- Categories will be loaded here -->
        </div>

        <!-- Food Items -->
        <div class="food-grid" id="foodGrid">
            <!-- Food items will be loaded here -->
        </div>

        <!-- Loading Overlay -->
        <div class="loading" id="loading">
            <div class="spinner"></div>
        </div>
    </div>

    <!-- Floating Cart Button -->
    <div class="floating-cart" id="floatingCart">
        🛒
        <div class="cart-count" id="cartCount">0</div>
    </div>

    <!-- Cart Drawer -->
    <div class="cart-container" id="cartContainer">
        <div class="cart-header">
            <div class="cart-title">Your Cart</div>
            <button class="close-cart" id="closeCart">×</button>
        </div>
        
        <div class="cart-items" id="cartItems">
            <!-- Cart items will be loaded here -->
        </div>
        
        <div class="cart-total">
            <div class="total-label">Total:</div>
            <div class="total-amount" id="cartTotal">0</div>
        </div>
        
        <button class="checkout-btn" id="checkoutBtn" disabled>
            Checkout
        </button>
    </div>

    <!-- Filter Modal -->
    <div class="filter-modal" id="filterModal">
        <div class="modal-content">
            <div class="modal-header">
                <div class="modal-title">Sort by</div>
                <button class="close-modal" id="closeFilterModal">×</button>
            </div>
            <div class="filter-options">
                <div class="filter-option" data-sort="rating">Rating (High to Low)</div>
                <div class="filter-option" data-sort="new">Newest</div>
                <div class="filter-option" data-sort="price_asc">Price (Low to High)</div>
                <div class="filter-option" data-sort="price_desc">Price (High to Low)</div>
            </div>
        </div>
    </div>

    <!-- Checkout Modal -->
    <div class="modal" id="checkoutModal">
        <div class="modal-content">
            <div class="modal-header">
                <div class="modal-title">Checkout</div>
                <button class="close-modal" id="closeCheckoutModal">×</button>
            </div>
            
            <form id="checkoutForm">
                <div class="form-group">
                    <label class="form-label">Full Name</label>
                    <input type="text" 
                           class="form-input" 
                           id="customerName"
                           required>
                </div>
                
                <div class="form-group">
                    <label class="form-label">Phone Number</label>
                    <input type="tel" 
                           class="form-input" 
                           id="phone"
                           placeholder="+998 __ ___ __ __"
                           required>
                </div>
                
                <div class="form-group">
                    <label class="form-label">Comments (Optional)</label>
                    <textarea class="form-input" 
                              id="comment" 
                              rows="3"
                              placeholder="Special instructions..."></textarea>
                </div>
                
                <div class="form-group">
                    <label class="form-label">Delivery Location</label>
                    <button type="button" class="location-btn" id="getLocationBtn">
                        📍 Get Current Location
                    </button>
                    <div class="location-coords" id="locationCoords">
                        Location: <span id="latLng"></span>
                    </div>
                </div>
                
                <div class="promo-section">
                    <input type="text" 
                           class="promo-input" 
                           id="promoCode"
                           placeholder="Promo code">
                    <button type="button" class="promo-btn" id="applyPromoBtn">
                        Apply
                    </button>
                </div>
                
                <div class="promo-success" id="promoSuccess"></div>
                <div class="promo-error" id="promoError"></div>
                
                <div class="form-group">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span>Subtotal:</span>
                        <span id="subtotalAmount">0</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span>Discount:</span>
                        <span id="discountAmount">0</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 18px; font-weight: 600;">
                        <span>Total:</span>
                        <span id="finalTotal">0</span>
                    </div>
                </div>
                
                <button type="submit" class="submit-order" id="submitOrderBtn" disabled>
                    Place Order
                </button>
            </form>
        </div>
    </div>

    <script>
        // Telegram WebApp
        const tg = window.Telegram.WebApp;
        tg.expand();
        tg.MainButton.hide();
        
        // State
        let cart = JSON.parse(localStorage.getItem('cart')) || {};
        let categories = [];
        let foods = [];
        let currentSort = 'rating';
        let currentCategory = null;
        let userLocation = null;
        let appliedPromo = null;
        
        // DOM Elements
        const foodGrid = document.getElementById('foodGrid');
        const categoriesContainer = document.getElementById('categoriesContainer');
        const cartCount = document.getElementById('cartCount');
        const cartItems = document.getElementById('cartItems');
        const cartTotal = document.getElementById('cartTotal');
        const floatingCart = document.getElementById('floatingCart');
        const cartContainer = document.getElementById('cartContainer');
        const closeCart = document.getElementById('closeCart');
        const checkoutBtn = document.getElementById('checkoutBtn');
        const searchInput = document.getElementById('searchInput');
        const filterBtn = document.getElementById('filterBtn');
        const filterModal = document.getElementById('filterModal');
        const closeFilterModal = document.getElementById('closeFilterModal');
        const checkoutModal = document.getElementById('checkoutModal');
        const closeCheckoutModal = document.getElementById('closeCheckoutModal');
        const getLocationBtn = document.getElementById('getLocationBtn');
        const locationCoords = document.getElementById('locationCoords');
        const latLng = document.getElementById('latLng');
        const promoCode = document.getElementById('promoCode');
        const applyPromoBtn = document.getElementById('applyPromoBtn');
        const promoSuccess = document.getElementById('promoSuccess');
        const promoError = document.getElementById('promoError');
        const subtotalAmount = document.getElementById('subtotalAmount');
        const discountAmount = document.getElementById('discountAmount');
        const finalTotal = document.getElementById('finalTotal');
        const submitOrderBtn = document.getElementById('submitOrderBtn');
        const checkoutForm = document.getElementById('checkoutForm');
        const loading = document.getElementById('loading');
        
        // Initialize
        document.addEventListener('DOMContentLoaded', async () => {
            await loadCategories();
            await loadFoods();
            updateCartUI();
            
            // Set user name from Telegram
            const user = tg.initDataUnsafe?.user;
            if (user) {
                document.getElementById('customerName').value = 
                    `${user.first_name} ${user.last_name || ''}`.trim();
            }
            
            // Load cart from localStorage
            loadCartFromStorage();
        });
        
        // Event Listeners
        floatingCart.addEventListener('click', () => {
            cartContainer.classList.add('active');
        });
        
        closeCart.addEventListener('click', () => {
            cartContainer.classList.remove('active');
        });
        
        checkoutBtn.addEventListener('click', () => {
            if (calculateTotal() < 50000) {
                alert('Minimum order amount is 50,000 сум');
                return;
            }
            openCheckoutModal();
        });
        
        filterBtn.addEventListener('click', () => {
            filterModal.classList.add('active');
        });
        
        closeFilterModal.addEventListener('click', () => {
            filterModal.classList.remove('active');
        });
        
        closeCheckoutModal.addEventListener('click', () => {
            checkoutModal.classList.remove('active');
        });
        
        searchInput.addEventListener('input', (e) => {
            filterFoods(e.target.value);
        });
        
        getLocationBtn.addEventListener('click', getLocation);
        applyPromoBtn.addEventListener('click', applyPromo);
        checkoutForm.addEventListener('submit', submitOrder);
        
        // Filter modal options
        document.querySelectorAll('.filter-option').forEach(option => {
            option.addEventListener('click', (e) => {
                document.querySelectorAll('.filter-option').forEach(opt => {
                    opt.classList.remove('active');
                });
                e.target.classList.add('active');
                currentSort = e.target.dataset.sort;
                sortFoods();
                filterModal.classList.remove('active');
            });
        });
        
        // Functions
        async function loadCategories() {
            showLoading();
            try {
                const initData = tg.initData;
                const response = await fetch(`/api/categories?init_data=${encodeURIComponent(initData)}`);
                if (!response.ok) throw new Error('Failed to load categories');
                categories = await response.json();
                renderCategories();
            } catch (error) {
                console.error('Error loading categories:', error);
            } finally {
                hideLoading();
            }
        }
        
        async function loadFoods() {
            showLoading();
            try {
                const initData = tg.initData;
                const response = await fetch(`/api/foods?init_data=${encodeURIComponent(initData)}`);
                if (!response.ok) throw new Error('Failed to load foods');
                foods = await response.json();
                sortFoods();
            } catch (error) {
                console.error('Error loading foods:', error);
            } finally {
                hideLoading();
            }
        }
        
        function renderCategories() {
            categoriesContainer.innerHTML = '';
            
            // Add "All" category
            const allBtn = document.createElement('button');
            allBtn.className = `category-btn ${!currentCategory ? 'active' : ''}`;
            allBtn.textContent = 'All';
            allBtn.addEventListener('click', () => {
                currentCategory = null;
                document.querySelectorAll('.category-btn').forEach(btn => {
                    btn.classList.remove('active');
                });
                allBtn.classList.add('active');
                filterFoods();
            });
            categoriesContainer.appendChild(allBtn);
            
            // Add other categories
            categories.forEach(category => {
                const btn = document.createElement('button');
                btn.className = `category-btn ${currentCategory === category.id ? 'active' : ''}`;
                btn.textContent = category.name;
                btn.dataset.categoryId = category.id;
                btn.addEventListener('click', () => {
                    currentCategory = category.id;
                    document.querySelectorAll('.category-btn').forEach(btn => {
                        btn.classList.remove('active');
                    });
                    btn.classList.add('active');
                    filterFoods();
                });
                categoriesContainer.appendChild(btn);
            });
        }
        
        function renderFoods(filteredFoods) {
            foodGrid.innerHTML = '';
            
            if (filteredFoods.length === 0) {
                foodGrid.innerHTML = '<div style="text-align: center; color: var(--text-secondary); padding: 40px;">No food items found</div>';
                return;
            }
            
            filteredFoods.forEach(food => {
                const foodCard = document.createElement('div');
                foodCard.className = 'food-card';
                
                const inCart = cart[food.id] || 0;
                
                foodCard.innerHTML = `
                    ${food.image_url ? `<img src="${food.image_url}" alt="${food.name}" class="food-image">` : ''}
                    <div class="food-info">
                        <div class="food-header">
                            <div class="food-name">${food.name}</div>
                            <div class="food-rating">
                                ⭐ ${food.rating.toFixed(1)}
                                ${food.is_new ? '<span style="color: var(--accent); margin-left: 8px;">NEW</span>' : ''}
                            </div>
                        </div>
                        <div class="food-desc">${food.description || ''}</div>
                        <div class="food-footer">
                            <div class="food-price">${food.price.toLocaleString()} сум</div>
                            <button class="add-btn" data-food-id="${food.id}">
                                ${inCart > 0 ? `${inCart} in cart` : 'Add'}
                            </button>
                        </div>
                    </div>
                `;
                
                foodGrid.appendChild(foodCard);
                
                // Add event listener to the button
                const addBtn = foodCard.querySelector('.add-btn');
                addBtn.addEventListener('click', () => {
                    addToCart(food.id, food.name, food.price);
                });
            });
        }
        
        function filterFoods(searchTerm = '') {
            let filtered = foods;
            
            // Filter by category
            if (currentCategory) {
                filtered = filtered.filter(food => food.category_id === currentCategory);
            }
            
            // Filter by search term
            if (searchTerm) {
                const term = searchTerm.toLowerCase();
                filtered = filtered.filter(food => 
                    food.name.toLowerCase().includes(term) ||
                    (food.description && food.description.toLowerCase().includes(term))
                );
            }
            
            sortFoods(filtered);
        }
        
        function sortFoods(foodsToSort = foods) {
            let sorted = [...foodsToSort];
            
            switch (currentSort) {
                case 'rating':
                    sorted.sort((a, b) => b.rating - a.rating);
                    break;
                case 'new':
                    sorted.sort((a, b) => (b.is_new ? 1 : 0) - (a.is_new ? 1 : 0));
                    break;
                case 'price_asc':
                    sorted.sort((a, b) => a.price - b.price);
                    break;
                case 'price_desc':
                    sorted.sort((a, b) => b.price - a.price);
                    break;
            }
            
            renderFoods(sorted);
        }
        
        function addToCart(foodId, foodName, foodPrice) {
            if (!cart[foodId]) {
                cart[foodId] = 0;
            }
            cart[foodId]++;
            
            saveCartToStorage();
            updateCartUI();
            updateFoodButton(foodId);
        }
        
        function removeFromCart(foodId) {
            if (cart[foodId]) {
                cart[foodId]--;
                if (cart[foodId] <= 0) {
                    delete cart[foodId];
                }
            }
            
            saveCartToStorage();
            updateCartUI();
            updateFoodButton(foodId);
        }
        
        function updateFoodButton(foodId) {
            const addBtn = document.querySelector(`.add-btn[data-food-id="${foodId}"]`);
            if (addBtn) {
                addBtn.textContent = cart[foodId] > 0 ? `${cart[foodId]} in cart` : 'Add';
            }
        }
        
        function updateCartUI() {
            const totalItems = Object.values(cart).reduce((sum, qty) => sum + qty, 0);
            const totalAmount = calculateTotal();
            
            cartCount.textContent = totalItems;
            cartTotal.textContent = totalAmount.toLocaleString() + ' сум';
            
            checkoutBtn.disabled = totalAmount < 50000;
            
            // Update cart items list
            cartItems.innerHTML = '';
            
            if (totalItems === 0) {
                cartItems.innerHTML = '<div style="text-align: center; color: var(--text-secondary); padding: 20px;">Cart is empty</div>';
                return;
            }
            
            Object.keys(cart).forEach(foodId => {
                const food = foods.find(f => f.id == foodId);
                if (!food) return;
                
                const qty = cart[foodId];
                const itemTotal = food.price * qty;
                
                const cartItem = document.createElement('div');
                cartItem.className = 'cart-item';
                cartItem.innerHTML = `
                    <div class="cart-item-info">
                        <div class="cart-item-name">${food.name}</div>
                        <div class="cart-item-price">${food.price.toLocaleString()} сум × ${qty}</div>
                    </div>
                    <div class="cart-item-controls">
                        <button class="qty-btn" data-food-id="${foodId}" data-action="remove">-</button>
                        <span class="cart-item-qty">${qty}</span>
                        <button class="qty-btn" data-food-id="${foodId}" data-action="add">+</button>
                    </div>
                `;
                
                cartItems.appendChild(cartItem);
            });
            
            // Add event listeners to quantity buttons
            cartItems.querySelectorAll('.qty-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const foodId = e.target.dataset.foodId;
                    const action = e.target.dataset.action;
                    
                    if (action === 'add') {
                        addToCart(foodId);
                    } else if (action === 'remove') {
                        removeFromCart(foodId);
                    }
                });
            });
        }
        
        function calculateTotal() {
            let total = 0;
            Object.keys(cart).forEach(foodId => {
                const food = foods.find(f => f.id == foodId);
                if (food) {
                    total += food.price * cart[foodId];
                }
            });
            return total;
        }
        
        function saveCartToStorage() {
            localStorage.setItem('cart', JSON.stringify(cart));
        }
        
        function loadCartFromStorage() {
            const saved = localStorage.getItem('cart');
            if (saved) {
                cart = JSON.parse(saved);
                updateCartUI();
            }
        }
        
        function openCheckoutModal() {
            const total = calculateTotal();
            subtotalAmount.textContent = total.toLocaleString() + ' сум';
            
            // Reset promo
            appliedPromo = null;
            promoSuccess.style.display = 'none';
            promoError.style.display = 'none';
            promoCode.value = '';
            updateCheckoutTotal();
            
            checkoutModal.classList.add('active');
        }
        
        function updateCheckoutTotal() {
            const subtotal = calculateTotal();
            let discount = 0;
            
            if (appliedPromo) {
                discount = Math.floor(subtotal * appliedPromo.discount_percent / 100);
            }
            
            const final = subtotal - discount;
            
            discountAmount.textContent = discount.toLocaleString() + ' сум';
            finalTotal.textContent = final.toLocaleString() + ' сум';
            
            submitOrderBtn.disabled = !userLocation || final < 50000;
        }
        
        function getLocation() {
            if (!navigator.geolocation) {
                alert('Geolocation is not supported by your browser');
                return;
            }
            
            getLocationBtn.textContent = 'Getting location...';
            getLocationBtn.disabled = true;
            
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    userLocation = {
                        lat: position.coords.latitude,
                        lng: position.coords.longitude
                    };
                    
                    latLng.textContent = `${userLocation.lat.toFixed(6)}, ${userLocation.lng.toFixed(6)}`;
                    locationCoords.classList.add('active');
                    
                    getLocationBtn.textContent = '📍 Location received';
                    updateCheckoutTotal();
                },
                (error) => {
                    alert('Error getting location: ' + error.message);
                    getLocationBtn.textContent = '📍 Get Current Location';
                    getLocationBtn.disabled = false;
                }
            );
        }
        
        async function applyPromo() {
            const code = promoCode.value.trim();
            if (!code) return;
            
            promoSuccess.style.display = 'none';
            promoError.style.display = 'none';
            
            try {
                const initData = tg.initData;
                const response = await fetch(`/api/promo/validate?init_data=${encodeURIComponent(initData)}&code=${encodeURIComponent(code)}`, {
                    method: 'POST'
                });
                
                if (!response.ok) throw new Error('Validation failed');
                
                const result = await response.json();
                
                if (result.valid) {
                    appliedPromo = result;
                    promoSuccess.textContent = `Promo code applied! ${result.discount_percent}% discount`;
                    promoSuccess.style.display = 'block';
                    updateCheckoutTotal();
                } else {
                    promoError.textContent = result.message;
                    promoError.style.display = 'block';
                }
            } catch (error) {
                promoError.textContent = 'Error validating promo code';
                promoError.style.display = 'block';
            }
        }
        
        async function submitOrder(e) {
            e.preventDefault();
            
            if (!userLocation) {
                alert('Please get your location first');
                return;
            }
            
            showLoading();
            
            try {
                // Prepare order items
                const orderItems = [];
                Object.keys(cart).forEach(foodId => {
                    const food = foods.find(f => f.id == foodId);
                    if (food) {
                        orderItems.push({
                            food_id: food.id,
                            name: food.name,
                            qty: cart[foodId],
                            price: food.price
                        });
                    }
                });
                
                // Calculate total with discount
                const subtotal = calculateTotal();
                let discount = 0;
                if (appliedPromo) {
                    discount = Math.floor(subtotal * appliedPromo.discount_percent / 100);
                }
                const total = subtotal - discount;
                
                // Prepare order data
                const orderData = {
                    type: "order_create",
                    items: orderItems,
                    total: total,
                    subtotal: subtotal,
                    discount: discount,
                    promo_code: appliedPromo?.code || null,
                    customer_name: document.getElementById('customerName').value,
                    phone: document.getElementById('phone').value,
                    comment: document.getElementById('comment').value,
                    location: userLocation,
                    created_at_client: new Date().toISOString()
                };
                
                // Send to Telegram bot
                tg.sendData(JSON.stringify(orderData));
                
                // Clear cart
                cart = {};
                saveCartToStorage();
                updateCartUI();
                
                // Close modals
                checkoutModal.classList.remove('active');
                cartContainer.classList.remove('active');
                
                // Show success message
                alert('Order placed successfully! You will receive confirmation shortly.');
                
                // Close WebApp after delay
                setTimeout(() => {
                    tg.close();
                }, 2000);
                
            } catch (error) {
                console.error('Error submitting order:', error);
                alert('Error placing order. Please try again.');
            } finally {
                hideLoading();
            }
        }
        
        function showLoading() {
            loading.classList.add('active');
        }
        
        function hideLoading() {
            loading.classList.remove('active');
        }
        
        // Auto-close WebApp when user goes back
        tg.BackButton.onClick(() => {
            tg.close();
        });
    </script>
</body>
</html>
"""

# ============================================================================
# Bot Handlers - Client
# ============================================================================

@dp.message(Command("start"))
async def cmd_start(message: Message, db: AsyncSession = Depends(get_db)):
    """Handle /start command with referral"""
    args = message.text.split()
    ref_id = None
    
    # Check for referral
    if len(args) > 1:
        try:
            ref_id = int(args[1])
        except ValueError:
            pass
    
    # Check if user exists
    result = await db.execute(
        select(User).where(User.tg_id == message.from_user.id)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        # Create new user
        ref_by_user = None
        if ref_id:
            # Check if referrer exists and not self-referral
            if ref_id != message.from_user.id:
                result = await db.execute(
                    select(User).where(User.tg_id == ref_id)
                )
                ref_by_user = result.scalar_one_or_none()
        
        user = User(
            tg_id=message.from_user.id,
            username=message.from_user.username,
            full_name=message.from_user.full_name,
            ref_by_user_id=ref_by_user.id if ref_by_user else None
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
    
    # Send welcome message
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(text="🛍 Заказать", web_app=WebAppInfo(url=config.WEBAPP_URL))
            ],
            [
                KeyboardButton(text="📦 Мои заказы"),
                KeyboardButton(text="ℹ️ Информация о нас")
            ],
            [
                KeyboardButton(text="👥 Пригласить друга")
            ]
        ],
        resize_keyboard=True,
        input_field_placeholder="Выберите действие..."
    )
    
    await message.answer(
        f"Добро пожаловать в FIESTA! {message.from_user.full_name}\n\n"
        "Для заказа перейдите по кнопке ➡️\n"
        "🛍 Заказать",
        reply_markup=keyboard
    )

@dp.message(lambda message: message.text == "📦 Мои заказы")
async def my_orders(message: Message, db: AsyncSession = Depends(get_db)):
    """Show user's orders"""
    # Get user
    result = await db.execute(
        select(User).where(User.tg_id == message.from_user.id)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        await message.answer("Сначала начните с /start")
        return
    
    # Get orders
    result = await db.execute(
        select(Order)
        .where(Order.user_id == user.id)
        .order_by(Order.created_at.desc())
        .limit(10)
    )
    orders = result.scalars().all()
    
    if not orders:
        await message.answer(
            "В данный момент у вас нет активных заказов в нашем магазине.\n"
            "Чтобы открыть магазин, введите команду — /shop",
            reply_markup=ReplyKeyboardRemove()
        )
        return
    
    for order in orders:
        # Get order items
        result = await db.execute(
            select(OrderItem).where(OrderItem.order_id == order.id)
        )
        items = result.scalars().all()
        
        items_text = "\n".join([
            f"• {item.name_snapshot} x{item.qty} = {item.line_total:,} сум"
            for item in items
        ])
        
        status_text = {
            'NEW': 'Принят',
            'CONFIRMED': 'Подтвержден',
            'COOKING': 'Готовится',
            'COURIER_ASSIGNED': 'Курьер назначен',
            'OUT_FOR_DELIVERY': 'Передан курьеру',
            'DELIVERED': 'Доставлен',
            'CANCELED': 'Отменен'
        }.get(order.status, order.status)
        
        await message.answer(
            f"🆔 Заказ №{order.order_number}\n"
            f"📅 {order.created_at.strftime('%d.%m.%Y %H:%M')}\n"
            f"💰 {order.total:,} сум\n"
            f"📦 {status_text}\n\n"
            f"🍽️ Заказ:\n{items_text}"
        )

@dp.message(Command("shop"))
async def cmd_shop(message: Message):
    """Shop command"""
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🛍 Заказать",
                    web_app=WebAppInfo(url=config.WEBAPP_URL)
                )
            ]
        ]
    )
    
    await message.answer(
        "Чтобы открыть наш магазин, нажмите кнопку ниже",
        reply_markup=keyboard
    )

@dp.message(lambda message: message.text == "ℹ️ Информация о нас")
async def about_us(message: Message):
    """About us information"""
    await message.answer(
        "🌟 Добро Пожаловать в FIESTA !\n\n"
        "📍 Наш адрес:Хорезмская область, г.Хива, махаллинский сход граждан Гиламчи\n"
        "🏢 Ориентир: Школа №12 Оруджева\n"
        "📞 Контактный номер: +998 91 420 15 15\n"
        "🕙 Рабочие часы: 24/7\n"
        "📷 Мы в Instagram: fiesta.khiva (https://www.instagram.com/fiesta.khiva?igsh=Z3VoMzE0eGx0ZTVo)\n"
        "🔗 Найти нас на карте: Место расположение (https://maps.app.goo.gl/dpBVHBWX1K7NTYVR7)",
        disable_web_page_preview=True
    )

@dp.message(lambda message: message.text == "👥 Пригласить друга")
async def invite_friend(message: Message, db: AsyncSession = Depends(get_db)):
    """Referral system"""
    # Get user
    result = await db.execute(
        select(User).where(User.tg_id == message.from_user.id)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        await message.answer("Сначала начните с /start")
        return
    
    # Get referral stats
    # Count referrals
    result = await db.execute(
        select(func.count(User.id)).where(User.ref_by_user_id == user.id)
    )
    ref_count = result.scalar() or 0
    
    # Count user's orders
    result = await db.execute(
        select(func.count(Order.id)).where(Order.user_id == user.id)
    )
    orders_count = result.scalar() or 0
    
    # Count delivered orders
    result = await db.execute(
        select(func.count(Order.id)).where(
            Order.user_id == user.id,
            Order.status == 'DELIVERED'
        )
    )
    delivered_count = result.scalar() or 0
    
    # Check if user already has a promo for referrals
    result = await db.execute(
        select(Promo).where(
            Promo.code.like(f"REF_{user.tg_id}_%")
        )
    )
    existing_promo = result.scalar_one_or_none()
    
    # Create promo if ref_count >= 3 and no existing promo
    if ref_count >= 3 and not existing_promo:
        promo_code = f"REF_{user.tg_id}_{secrets.token_hex(3).upper()}"
        promo = Promo(
            code=promo_code,
            discount_percent=15,
            expires_at=None,  # Never expires
            usage_limit=5,
            is_active=True
        )
        db.add(promo)
        await db.commit()
        
        await message.answer(
            f"🎉 Поздравляем! Вы пригласили {ref_count} друзей.\n"
            f"Ваш промо-код на 15% скидку: {promo_code}\n"
            f"Лимит использований: 5 раз"
        )
    
    bot_username = config.get_bot_username()
    referral_link = f"https://t.me/{bot_username}?start={user.tg_id}"
    
    await message.answer(
        "За приглашение друга, вы можете получить промо-код от нас\n\n"
        f"👥 Вы пригласили {ref_count} человек\n"
        f"🛒 Оформили заказов: {orders_count}\n"
        f"💰 Оплатили заказов: {delivered_count}\n\n"
        f"👤 Ваша реферальная ссылка:\n{referral_link}\n\n"
        "Пригласите трех человек и вы получите от нас промо-код со скидкой 15%"
    )

# ============================================================================
# Bot Handlers - WebApp Data
# ============================================================================

@dp.message(lambda message: message.web_app_data is not None)
async def handle_webapp_data(message: Message, db: AsyncSession = Depends(get_db)):
    """Handle data from WebApp"""
    try:
        data = json.loads(message.web_app_data.data)
        
        if data.get("type") == "order_create":
            await handle_order_create(message, data, db)
    except Exception as e:
        logging.error(f"Error processing WebApp data: {e}")
        await message.answer("Ошибка при обработке заказа. Пожалуйста, попробуйте еще раз.")

async def handle_order_create(message: Message, data: dict, db: AsyncSession):
    """Create order from WebApp data"""
    # Validate total amount
    if data["total"] < config.MIN_ORDER_AMOUNT:
        await message.answer(
            f"Минимальная сумма заказа {config.MIN_ORDER_AMOUNT:,} сум. "
            f"Ваша сумма: {data['total']:,} сум"
        )
        return
    
    # Get or create user
    result = await db.execute(
        select(User).where(User.tg_id == message.from_user.id)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        user = User(
            tg_id=message.from_user.id,
            username=message.from_user.username,
            full_name=message.from_user.full_name
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
    
    # Check promo code
    promo_id = None
    if data.get("promo_code"):
        result = await db.execute(
            select(Promo).where(
                Promo.code == data["promo_code"],
                Promo.is_active == True,
                or_(
                    Promo.expires_at.is_(None),
                    Promo.expires_at > datetime.now(timezone.utc)
                )
            )
        )
        promo = result.scalar_one_or_none()
        
        if promo and (promo.usage_limit is None or promo.used_count < promo.usage_limit):
            promo_id = promo.id
            promo.used_count += 1
    
    # Generate order number
    order_number = f"ORD-{datetime.now().strftime('%Y%m%d')}-{secrets.token_hex(3).upper()}"
    
    # Create order
    order = Order(
        order_number=order_number,
        user_id=user.id,
        customer_name=data["customer_name"],
        phone=data["phone"],
        comment=data.get("comment"),
        total=data["total"],
        status="NEW",
        location_lat=data["location"]["lat"],
        location_lng=data["location"]["lng"],
        promo_id=promo_id
    )
    db.add(order)
    await db.commit()
    await db.refresh(order)
    
    # Create order items
    for item in data["items"]:
        order_item = OrderItem(
            order_id=order.id,
            food_id=item.get("food_id"),
            name_snapshot=item["name"],
            price_snapshot=item["price"],
            qty=item["qty"],
            line_total=item["price"] * item["qty"]
        )
        db.add(order_item)
    
    await db.commit()
    
    # Send confirmation to user
    await message.answer(
        f"Ваш заказ принят ✅\n\n"
        f"🆔 Заказ №{order.order_number}\n"
        f"💰 Сумма: {order.total:,} сум\n"
        f"📦 Статус: Принят\n\n"
        f"Мы скоро свяжемся с вами для подтверждения."
    )
    
    # Send to admin channel
    if config.SHOP_CHANNEL_ID:
        await send_order_to_admin_channel(order, user, db)

async def send_order_to_admin_channel(order: Order, user: User, db: AsyncSession):
    """Send order notification to admin channel"""
    # Get order items
    result = await db.execute(
        select(OrderItem).where(OrderItem.order_id == order.id)
    )
    items = result.scalars().all()
    
    items_text = "\n".join([
        f"{item.name_snapshot} x{item.qty} = {item.line_total:,} сум"
        for item in items
    ])
    
    # Create inline keyboard for admin actions
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="✅ Подтвержден",
                    callback_data=f"confirm_order:{order.id}"
                ),
                InlineKeyboardButton(
                    text="🍳 Готовится",
                    callback_data=f"cooking_order:{order.id}"
                )
            ],
            [
                InlineKeyboardButton(
                    text="🚴 Курьер",
                    callback_data=f"assign_courier:{order.id}"
                )
            ]
        ]
    )
    
    message_text = (
        f"🆕 Новый заказ №{order.order_number}\n\n"
        f"👤 Пользователь: {order.customer_name} (@{user.username if user.username else 'нет'})\n"
        f"📞 Телефон: {order.phone}\n"
        f"💰 Сумма: {order.total:,} сум\n"
        f"🕒 Время: {order.created_at.strftime('%d.%m.%Y %H:%M')}\n"
        f"📍 Локация: https://maps.google.com/?q={order.location_lat},{order.location_lng}\n"
        f"💬 Комментарий: {order.comment or 'нет'}\n\n"
        f"🍽️ Заказ:\n{items_text}"
    )
    
    try:
        await bot.send_message(
            chat_id=config.SHOP_CHANNEL_ID,
            text=message_text,
            reply_markup=keyboard
        )
    except Exception as e:
        logging.error(f"Error sending to admin channel: {e}")

# ============================================================================
# Bot Handlers - Admin
# ============================================================================

@dp.message(Command("admin"))
async def cmd_admin(message: Message):
    """Admin panel"""
    if message.from_user.id not in config.ADMIN_IDS:
        await message.answer("Доступ запрещен")
        return
    
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🍔 Taомлар", callback_data="admin_foods")],
            [InlineKeyboardButton(text="📂 Kategoriyalar", callback_data="admin_categories")],
            [InlineKeyboardButton(text="🎁 Promokodlar", callback_data="admin_promos")],
            [InlineKeyboardButton(text="📊 Statistika", callback_data="admin_stats")],
            [InlineKeyboardButton(text="🚴 Kuryerlar", callback_data="admin_couriers")],
            [InlineKeyboardButton(text="📦 Aktiv buyurtmalar", callback_data="admin_active_orders")],
            [InlineKeyboardButton(text="⚙️ Sozlamalar", callback_data="admin_settings")]
        ]
    )
    
    await message.answer(
        "⚙️ Админ панель\n\n"
        "Выберите раздел:",
        reply_markup=keyboard
    )

@dp.callback_query(lambda c: c.data.startswith("admin_"))
async def admin_callback_handler(callback: CallbackQuery, db: AsyncSession = Depends(get_db)):
    """Handle admin callbacks"""
    if callback.from_user.id not in config.ADMIN_IDS:
        await callback.answer("Доступ запрещен")
        return
    
    action = callback.data
    
    if action == "admin_foods":
        await show_foods_admin(callback, db)
    elif action == "admin_categories":
        await show_categories_admin(callback, db)
    elif action == "admin_promos":
        await show_promos_admin(callback, db)
    elif action == "admin_stats":
        await show_stats_admin(callback, db)
    elif action == "admin_couriers":
        await show_couriers_admin(callback, db)
    elif action == "admin_active_orders":
        await show_active_orders_admin(callback, db)
    elif action == "admin_settings":
        await show_settings_admin(callback, db)

async def show_active_orders_admin(callback: CallbackQuery, db: AsyncSession):
    """Show active orders for admin"""
    # Get active orders (not DELIVERED or CANCELED)
    result = await db.execute(
        select(Order)
        .where(
            Order.status.in_(["NEW", "CONFIRMED", "COOKING", "COURIER_ASSIGNED", "OUT_FOR_DELIVERY"])
        )
        .order_by(Order.created_at.desc())
        .limit(20)
    )
    orders = result.scalars().all()
    
    if not orders:
        await callback.message.edit_text("Активных заказов нет")
        return
    
    keyboard_buttons = []
    for order in orders:
        status_text = {
            'NEW': '🆕',
            'CONFIRMED': '✅',
            'COOKING': '🍳',
            'COURIER_ASSIGNED': '🚴',
            'OUT_FOR_DELIVERY': '📦'
        }.get(order.status, order.status)
        
        keyboard_buttons.append([
            InlineKeyboardButton(
                text=f"{status_text} #{order.order_number} - {order.total:,} сум",
                callback_data=f"view_order:{order.id}"
            )
        ])
    
    keyboard_buttons.append([
        InlineKeyboardButton(text="🔙 Назад", callback_data="admin_back")
    ])
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_buttons)
    
    await callback.message.edit_text(
        f"📦 Активные заказы ({len(orders)}):",
        reply_markup=keyboard
    )

# ============================================================================
# Bot Handlers - Order Status Updates
# ============================================================================

@dp.callback_query(lambda c: c.data.startswith("confirm_order:"))
async def confirm_order(callback: CallbackQuery, db: AsyncSession = Depends(get_db)):
    """Confirm order"""
    if callback.from_user.id not in config.ADMIN_IDS:
        await callback.answer("Доступ запрещен")
        return
    
    order_id = int(callback.data.split(":")[1])
    
    # Update order status
    result = await db.execute(
        update(Order)
        .where(Order.id == order_id)
        .values(status="CONFIRMED", updated_at=datetime.now(timezone.utc))
        .returning(Order)
    )
    order = result.scalar_one_or_none()
    
    if order:
        await db.commit()
        
        # Notify user
        try:
            await bot.send_message(
                chat_id=order.user.tg_id,
                text=f"✅ Ваш заказ №{order.order_number} подтвержден и готовится!"
            )
        except Exception:
            pass
        
        # Update admin message
        await callback.message.edit_text(
            callback.message.text + "\n\n✅ Подтвержден администратором",
            reply_markup=None
        )
        
        await callback.answer("Заказ подтвержден")

@dp.callback_query(lambda c: c.data.startswith("assign_courier:"))
async def assign_courier(callback: CallbackQuery, db: AsyncSession = Depends(get_db)):
    """Assign courier to order"""
    if callback.from_user.id not in config.ADMIN_IDS:
        await callback.answer("Доступ запрещен")
        return
    
    order_id = int(callback.data.split(":")[1])
    
    # Get active couriers
    result = await db.execute(
        select(Courier).where(Courier.is_active == True)
    )
    couriers = result.scalars().all()
    
    if not couriers:
        await callback.answer("Нет активных курьеров")
        return
    
    # Create courier selection keyboard
    keyboard_buttons = []
    for courier in couriers:
        keyboard_buttons.append([
            InlineKeyboardButton(
                text=f"🚴 {courier.name}",
                callback_data=f"select_courier:{order_id}:{courier.id}"
            )
        ])
    
    keyboard_buttons.append([
        InlineKeyboardButton(text="🔙 Назад", callback_data="admin_back")
    ])
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_buttons)
    
    await callback.message.edit_text(
        f"Выберите курьера для заказа №{order_id}:",
        reply_markup=keyboard
    )

@dp.callback_query(lambda c: c.data.startswith("select_courier:"))
async def select_courier_handler(callback: CallbackQuery, db: AsyncSession = Depends(get_db)):
    """Handle courier selection"""
    if callback.from_user.id not in config.ADMIN_IDS:
        await callback.answer("Доступ запрещен")
        return
    
    _, order_id, courier_id = callback.data.split(":")
    order_id = int(order_id)
    courier_id = int(courier_id)
    
    # Update order
    result = await db.execute(
        update(Order)
        .where(Order.id == order_id)
        .values(
            status="COURIER_ASSIGNED",
            courier_id=courier_id,
            updated_at=datetime.now(timezone.utc)
        )
        .returning(Order)
    )
    order = result.scalar_one_or_none()
    
    if order:
        await db.commit()
        
        # Get courier
        result = await db.execute(
            select(Courier).where(Courier.id == courier_id)
        )
        courier = result.scalar_one_or_none()
        
        # Notify courier
        if courier and config.COURIER_CHANNEL_ID:
            await notify_courier(order, courier, db)
        
        # Update admin message
        await callback.message.edit_text(
            callback.message.text + f"\n\n🚴 Назначен курьер: {courier.name}",
            reply_markup=None
        )
        
        await callback.answer("Курьер назначен")

async def notify_courier(order: Order, courier: Courier, db: AsyncSession):
    """Send order notification to courier"""
    # Get order items
    result = await db.execute(
        select(OrderItem).where(OrderItem.order_id == order.id)
    )
    items = result.scalars().all()
    
    items_text = "\n".join([
        f"• {item.name_snapshot} x{item.qty}"
        for item in items
    ])
    
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="✅ Qabul qildim",
                    callback_data=f"courier_accept:{order.id}"
                ),
                InlineKeyboardButton(
                    text="📦 Yetkazildi",
                    callback_data=f"courier_delivered:{order.id}"
                )
            ]
        ]
    )
    
    message_text = (
        f"🚴 Новый заказ №{order.order_number}\n\n"
        f"👤 Клиент: {order.customer_name}\n"
        f"📞 Телефон: {order.phone}\n"
        f"💰 Сумма: {order.total:,} сум\n"
        f"📍 Локация: https://maps.google.com/?q={order.location_lat},{order.location_lng}\n"
        f"💬 Комментарий: {order.comment or 'нет'}\n\n"
        f"🍽️ Список:\n{items_text}"
    )
    
    try:
        if config.COURIER_CHANNEL_ID:
            await bot.send_message(
                chat_id=config.COURIER_CHANNEL_ID,
                text=message_text,
                reply_markup=keyboard
            )
        else:
            await bot.send_message(
                chat_id=courier.chat_id,
                text=message_text,
                reply_markup=keyboard
            )
    except Exception as e:
        logging.error(f"Error notifying courier: {e}")

# ============================================================================
# Bot Handlers - Courier Actions
# ============================================================================

@dp.callback_query(lambda c: c.data.startswith("courier_accept:"))
async def courier_accept(callback: CallbackQuery, db: AsyncSession = Depends(get_db)):
    """Courier accepts order"""
    order_id = int(callback.data.split(":")[1])
    
    # Check if user is a courier
    result = await db.execute(
        select(Courier).where(Courier.chat_id == callback.from_user.id)
    )
    courier = result.scalar_one_or_none()
    
    if not courier:
        await callback.answer("Вы не курьер")
        return
    
    # Update order status
    result = await db.execute(
        update(Order)
        .where(Order.id == order_id, Order.courier_id == courier.id)
        .values(status="OUT_FOR_DELIVERY", updated_at=datetime.now(timezone.utc))
        .returning(Order)
    )
    order = result.scalar_one_or_none()
    
    if order:
        await db.commit()
        
        # Notify customer
        try:
            await bot.send_message(
                chat_id=order.user.tg_id,
                text=f"🚴 Ваш заказ №{order.order_number} передан курьеру и скоро будет доставлен!"
            )
        except Exception:
            pass
        
        # Update courier message
        await callback.message.edit_text(
            callback.message.text + "\n\n✅ Принято курьером",
            reply_markup=None
        )
        
        # Update admin channel message if exists
        if config.SHOP_CHANNEL_ID:
            try:
                # You might want to edit the original message or send update
                pass
            except Exception:
                pass
        
        await callback.answer("Заказ принят")

@dp.callback_query(lambda c: c.data.startswith("courier_delivered:"))
async def courier_delivered(callback: CallbackQuery, db: AsyncSession = Depends(get_db)):
    """Courier marks order as delivered"""
    order_id = int(callback.data.split(":")[1])
    
    # Check if user is a courier
    result = await db.execute(
        select(Courier).where(Courier.chat_id == callback.from_user.id)
    )
    courier = result.scalar_one_or_none()
    
    if not courier:
        await callback.answer("Вы не курьер")
        return
    
    # Update order status
    result = await db.execute(
        update(Order)
        .where(Order.id == order_id, Order.courier_id == courier.id)
        .values(
            status="DELIVERED",
            delivered_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        .returning(Order)
    )
    order = result.scalar_one_or_none()
    
    if order:
        await db.commit()
        
        # Notify customer
        try:
            await bot.send_message(
                chat_id=order.user.tg_id,
                text=f"🎉 Ваш заказ №{order.order_number} успешно доставлен! Спасибо за заказ!"
            )
        except Exception:
            pass
        
        # Update courier message
        await callback.message.edit_text(
            callback.message.text + "\n\n📦 Доставлен",
            reply_markup=None
        )
        
        # Update admin channel message if exists
        if config.SHOP_CHANNEL_ID:
            try:
                # Mark as delivered in admin channel
                pass
            except Exception:
                pass
        
        await callback.answer("Заказ доставлен")

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run FastAPI server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
