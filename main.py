#!/usr/bin/env python3
"""
Telegram Food Delivery System - Complete Backend for Railway
Production-ready system with your credentials
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

import asyncpg
from aiogram import Bot, Dispatcher, F, Router, html
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode, ChatType
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.redis import RedisStorage
from aiogram.types import (
    Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton,
    WebAppInfo, ReplyKeyboardMarkup, KeyboardButton,
    ReplyKeyboardRemove, WebAppData
)
from aiogram.utils.keyboard import InlineKeyboardBuilder, ReplyKeyboardBuilder
from pydantic import BaseModel
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager
import hashlib
import hmac

# ============================
# Configuration - Your Credentials
# ============================

@dataclass
class Config:
    BOT_TOKEN: str = "7917271389:AAE4PXCowGo6Bsfdy3Hrz3x689MLJdQmVi4"
    ADMIN_IDS: List[int] = [6365371142]
    DB_URL: str = "postgresql+asyncpg://postgres:BDAaILJKOITNLlMOjJNfWiRPbICwEcpZ@centerbeam.proxy.rlwy.net:35489/railway"
    REDIS_URL: str = "redis://default:GBrZNeUKJfqRlPcQUoUICWQpbQRtRRJp@ballast.proxy.rlwy.net:35411"
    SHOP_CHANNEL_ID: str = "-1003530497437"
    COURIER_CHANNEL_ID: str = "-1003707946746"
    WEBAPP_URL: str = "https://mainsufooduz.netlify.app"
    PORT: int = int(os.getenv("PORT", 8000))
    
    def __post_init__(self):
        # Convert Redis URL for asyncpg compatibility
        self.DB_URL = self.DB_URL.replace("postgresql+asyncpg://", "postgresql://")
        self.BOT_USERNAME: str = ""

config = Config()

# ============================
# Database Models
# ============================

class OrderStatus(str, Enum):
    NEW = "NEW"
    CONFIRMED = "CONFIRMED"
    COOKING = "COOKING"
    COURIER_ASSIGNED = "COURIER_ASSIGNED"
    OUT_FOR_DELIVERY = "OUT_FOR_DELIVERY"
    DELIVERED = "DELIVERED"
    CANCELED = "CANCELED"

class WebAppOrderItem(BaseModel):
    food_id: int
    name: str
    qty: int
    price: float

class WebAppOrderData(BaseModel):
    type: str = "order_create"
    items: List[WebAppOrderItem]
    total: float
    customer_name: str
    phone: str
    comment: Optional[str] = ""
    location: Dict[str, float]
    created_at_client: str

# ============================
# Database Layer
# ============================

class Database:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None
    
    async def connect(self):
        self.pool = await asyncpg.create_pool(self.connection_string, min_size=1, max_size=10)
        await self.init_db()
    
    async def init_db(self):
        async with self.pool.acquire() as conn:
            # Users table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    tg_id BIGINT UNIQUE NOT NULL,
                    username VARCHAR(255),
                    full_name VARCHAR(255) NOT NULL,
                    joined_at TIMESTAMP DEFAULT NOW(),
                    ref_by_user_id INTEGER REFERENCES users(id)
                )
            ''')
            
            # Categories table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS categories (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE
                )
            ''')
            
            # Foods table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS foods (
                    id SERIAL PRIMARY KEY,
                    category_id INTEGER REFERENCES categories(id),
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    price DECIMAL(10, 2) NOT NULL,
                    rating DECIMAL(3, 2) DEFAULT 5.0,
                    is_new BOOLEAN DEFAULT FALSE,
                    is_active BOOLEAN DEFAULT TRUE,
                    image_url TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            ''')
            
            # Couriers table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS couriers (
                    id SERIAL PRIMARY KEY,
                    chat_id BIGINT UNIQUE NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            ''')
            
            # Orders table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS orders (
                    id SERIAL PRIMARY KEY,
                    order_number VARCHAR(50) UNIQUE NOT NULL,
                    user_id INTEGER REFERENCES users(id),
                    customer_name VARCHAR(255) NOT NULL,
                    phone VARCHAR(50) NOT NULL,
                    comment TEXT,
                    total DECIMAL(10, 2) NOT NULL,
                    status VARCHAR(50) DEFAULT 'NEW',
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    delivered_at TIMESTAMP,
                    location_lat DECIMAL(9, 6),
                    location_lng DECIMAL(9, 6),
                    courier_id INTEGER REFERENCES couriers(id),
                    promo_id INTEGER
                )
            ''')
            
            # Order items table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS order_items (
                    id SERIAL PRIMARY KEY,
                    order_id INTEGER REFERENCES orders(id),
                    food_id INTEGER REFERENCES foods(id),
                    name_snapshot VARCHAR(255) NOT NULL,
                    price_snapshot DECIMAL(10, 2) NOT NULL,
                    qty INTEGER NOT NULL,
                    line_total DECIMAL(10, 2) NOT NULL
                )
            ''')
            
            # Promo codes table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS promos (
                    id SERIAL PRIMARY KEY,
                    code VARCHAR(50) UNIQUE NOT NULL,
                    discount_percent INTEGER NOT NULL CHECK (discount_percent BETWEEN 1 AND 90),
                    expires_at TIMESTAMP,
                    usage_limit INTEGER DEFAULT NULL,
                    used_count INTEGER DEFAULT 0,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            ''')
            
            # Insert sample categories if empty
            categories_count = await conn.fetchval("SELECT COUNT(*) FROM categories")
            if categories_count == 0:
                sample_categories = [
                    ("Lavash",),
                    ("Burger",),
                    ("Xaggi",),
                    ("Shaurma",),
                    ("Hotdog",),
                    ("Combo",),
                    ("Sneki",),
                    ("Sous",),
                    ("Napitki",)
                ]
                await conn.executemany(
                    "INSERT INTO categories (name) VALUES ($1)",
                    sample_categories
                )
            
            # Insert sample foods if empty
            foods_count = await conn.fetchval("SELECT COUNT(*) FROM foods")
            if foods_count == 0:
                sample_foods = [
                    (1, "Лаваш с говядиной", "Свежая лепешка с говядиной и овощами", 28000.00, 4.8, True),
                    (1, "Лаваш с курицей", "Свежая лепешка с курицей и овощами", 26000.00, 4.7, False),
                    (1, "Лаваш острый", "Свежая лепешка с острым мясом", 30000.00, 4.9, True),
                    (2, "Чизбургер", "Бургер с сыром и говядиной", 32000.00, 4.6, False),
                    (2, "Гамбургер", "Классический бургер", 25000.00, 4.5, False),
                    (2, "Биг Бургер", "Большой бургер с двойным мясом", 45000.00, 4.9, True),
                    (3, "Хагги классический", "Традиционный хагги", 35000.00, 4.7, False),
                    (4, "Шаурма говяжья", "Шаурма с говядиной", 22000.00, 4.8, False),
                    (4, "Шаурма куриная", "Шаурма с курицей", 20000.00, 4.6, False),
                    (5, "Хот-дог классический", "Хот-дог с сосиской", 15000.00, 4.5, False),
                    (6, "Комбо №1", "Бургер + картофель + напиток", 55000.00, 4.9, True),
                    (7, "Картофель фри", "Хрустящий картофель", 12000.00, 4.4, False),
                    (8, "Соус чесночный", "Чесночный соус", 3000.00, 4.8, False),
                    (9, "Coca-Cola 0.5л", "Газированный напиток", 8000.00, 4.3, False),
                ]
                await conn.executemany(
                    """INSERT INTO foods (category_id, name, description, price, rating, is_new) 
                    VALUES ($1, $2, $3, $4, $5, $6)""",
                    sample_foods
                )
    
    async def get_user(self, tg_id: int):
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(
                "SELECT * FROM users WHERE tg_id = $1",
                tg_id
            )
    
    async def create_user(self, tg_id: int, username: str, full_name: str, ref_by: Optional[int] = None):
        async with self.pool.acquire() as conn:
            return await conn.fetchrow('''
                INSERT INTO users (tg_id, username, full_name, ref_by_user_id)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (tg_id) DO UPDATE SET
                    username = EXCLUDED.username,
                    full_name = EXCLUDED.full_name
                RETURNING *
            ''', tg_id, username, full_name, ref_by)
    
    async def get_categories(self):
        async with self.pool.acquire() as conn:
            return await conn.fetch(
                "SELECT * FROM categories WHERE is_active = TRUE ORDER BY name"
            )
    
    async def get_foods(self, category_id: Optional[int] = None):
        async with self.pool.acquire() as conn:
            if category_id:
                return await conn.fetch('''
                    SELECT f.*, c.name as category_name 
                    FROM foods f
                    LEFT JOIN categories c ON f.category_id = c.id
                    WHERE f.is_active = TRUE 
                    AND (f.category_id = $1 OR $1 IS NULL)
                    ORDER BY f.name
                ''', category_id)
            else:
                return await conn.fetch('''
                    SELECT f.*, c.name as category_name 
                    FROM foods f
                    LEFT JOIN categories c ON f.category_id = c.id
                    WHERE f.is_active = TRUE 
                    ORDER BY f.name
                ''')
    
    async def get_food_by_id(self, food_id: int):
        async with self.pool.acquire() as conn:
            return await conn.fetchrow('''
                SELECT * FROM foods WHERE id = $1 AND is_active = TRUE
            ''', food_id)
    
    async def create_order(self, data: WebAppOrderData, user_id: int, promo_id: Optional[int] = None):
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Generate order number
                date_str = datetime.now().strftime('%Y%m%d')
                unique_id = uuid.uuid4().hex[:6].upper()
                order_number = f"ORD-{date_str}-{unique_id}"
                
                # Create order
                order = await conn.fetchrow('''
                    INSERT INTO orders (
                        order_number, user_id, customer_name, phone, comment,
                        total, status, location_lat, location_lng, promo_id
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    RETURNING *
                ''', order_number, user_id, data.customer_name, data.phone,
                    data.comment, data.total, OrderStatus.NEW.value,
                    data.location['lat'], data.location['lng'], promo_id)
                
                # Create order items
                for item in data.items:
                    await conn.execute('''
                        INSERT INTO order_items (
                            order_id, food_id, name_snapshot, price_snapshot, qty, line_total
                        ) VALUES ($1, $2, $3, $4, $5, $6)
                    ''', order['id'], item.food_id, item.name, item.price, item.qty, item.price * item.qty)
                
                return order
    
    async def get_user_orders(self, user_id: int, limit: int = 10):
        async with self.pool.acquire() as conn:
            return await conn.fetch('''
                SELECT * FROM orders 
                WHERE user_id = $1 
                ORDER BY created_at DESC 
                LIMIT $2
            ''', user_id, limit)
    
    async def get_order_by_id(self, order_id: int):
        async with self.pool.acquire() as conn:
            return await conn.fetchrow('''
                SELECT o.*, u.tg_id as user_tg_id, u.username, u.full_name as user_full_name,
                       c.name as courier_name
                FROM orders o
                LEFT JOIN users u ON o.user_id = u.id
                LEFT JOIN couriers c ON o.courier_id = c.id
                WHERE o.id = $1
            ''', order_id)
    
    async def get_order_items(self, order_id: int):
        async with self.pool.acquire() as conn:
            return await conn.fetch('''
                SELECT * FROM order_items WHERE order_id = $1
            ''', order_id)
    
    async def get_active_orders(self):
        async with self.pool.acquire() as conn:
            return await conn.fetch('''
                SELECT o.*, u.username, u.full_name as user_full_name 
                FROM orders o
                LEFT JOIN users u ON o.user_id = u.id
                WHERE o.status NOT IN ('DELIVERED', 'CANCELED')
                ORDER BY o.created_at DESC
            ''')
    
    async def update_order_status(self, order_id: int, status: OrderStatus, courier_id: Optional[int] = None):
        async with self.pool.acquire() as conn:
            updates = []
            params = [status.value, datetime.now()]
            param_count = 2
            
            if courier_id:
                updates.append(f"courier_id = ${param_count + 1}")
                params.append(courier_id)
                param_count += 1
            
            if status == OrderStatus.DELIVERED:
                updates.append(f"delivered_at = ${param_count + 1}")
                params.append(datetime.now())
                param_count += 1
            
            update_clause = ", ".join(updates) if updates else ""
            if update_clause:
                update_clause = ", " + update_clause
            
            params.append(order_id)
            
            await conn.execute(f'''
                UPDATE orders 
                SET status = $1, updated_at = $2{update_clause}
                WHERE id = ${param_count + 1}
            ''', *params)
    
    async def get_couriers(self, active_only: bool = True):
        async with self.pool.acquire() as conn:
            if active_only:
                return await conn.fetch(
                    "SELECT * FROM couriers WHERE is_active = TRUE ORDER BY name"
                )
            return await conn.fetch("SELECT * FROM couriers ORDER BY name")
    
    async def get_courier_by_id(self, courier_id: int):
        async with self.pool.acquire() as conn:
            return await conn.fetchrow('''
                SELECT * FROM couriers WHERE id = $1
            ''', courier_id)
    
    async def create_courier(self, chat_id: int, name: str):
        async with self.pool.acquire() as conn:
            return await conn.fetchrow('''
                INSERT INTO couriers (chat_id, name)
                VALUES ($1, $2)
                ON CONFLICT (chat_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    is_active = TRUE
                RETURNING *
            ''', chat_id, name)
    
    async def get_referral_stats(self, user_id: int):
        async with self.pool.acquire() as conn:
            # Get referral count
            ref_count = await conn.fetchval('''
                SELECT COUNT(*) FROM users 
                WHERE ref_by_user_id = $1
            ''', user_id)
            
            # Get user's orders count
            orders_count = await conn.fetchval('''
                SELECT COUNT(*) FROM orders 
                WHERE user_id = $1
            ''', user_id)
            
            # Get delivered orders count
            delivered_count = await conn.fetchval('''
                SELECT COUNT(*) FROM orders 
                WHERE user_id = $1 AND status = 'DELIVERED'
            ''', user_id)
            
            return {
                'ref_count': ref_count or 0,
                'orders_count': orders_count or 0,
                'delivered_count': delivered_count or 0
            }
    
    async def validate_promo(self, code: str):
        async with self.pool.acquire() as conn:
            promo = await conn.fetchrow('''
                SELECT * FROM promos 
                WHERE code = $1 
                AND is_active = TRUE 
                AND (expires_at IS NULL OR expires_at > NOW())
                AND (usage_limit IS NULL OR used_count < usage_limit)
            ''', code)
            return promo
    
    async def use_promo(self, promo_id: int):
        async with self.pool.acquire() as conn:
            await conn.execute('''
                UPDATE promos 
                SET used_count = used_count + 1 
                WHERE id = $1
            ''', promo_id)
    
    async def get_stats(self, period_days: int = 30):
        async with self.pool.acquire() as conn:
            since_date = datetime.now() - timedelta(days=period_days)
            
            orders_count = await conn.fetchval('''
                SELECT COUNT(*) FROM orders 
                WHERE created_at >= $1
            ''', since_date)
            
            delivered_count = await conn.fetchval('''
                SELECT COUNT(*) FROM orders 
                WHERE status = 'DELIVERED' AND delivered_at >= $1
            ''', since_date)
            
            revenue = await conn.fetchval('''
                SELECT COALESCE(SUM(total), 0) FROM orders 
                WHERE status = 'DELIVERED' AND delivered_at >= $1
            ''', since_date)
            
            return {
                'orders_count': orders_count or 0,
                'delivered_count': delivered_count or 0,
                'revenue': revenue or 0
            }

# ============================
# Services
# ============================

class OrderService:
    def __init__(self, db: Database, bot: Bot):
        self.db = db
        self.bot = bot
    
    async def create_order_from_webapp(self, data: WebAppOrderData, user_id: int):
        # Validate total
        if data.total < 50000:
            raise ValueError("Минимальная сумма заказа 50,000 сум")
        
        # Create order
        order = await self.db.create_order(data, user_id)
        
        # Get user info
        user = await self.db.get_user(user_id)
        
        # Format order items text
        items_text = "\n".join([
            f"• {item.name} x{item.qty} = {item.price * item.qty:,.0f} сум"
            for item in data.items
        ])
        
        # Send confirmation to user
        await self.bot.send_message(
            chat_id=user_id,
            text=f"""✅ Ваш заказ принят!

🆔 Заказ №{order['order_number']}
💰 Сумма: {data.total:,.0f} сум
📦 Статус: Принят

Ожидайте подтверждения от администратора.""",
            parse_mode=ParseMode.HTML
        )
        
        # Send to admin channel
        location_text = f"{data.location['lat']:.6f},{data.location['lng']:.6f}"
        location_url = f"https://maps.google.com/?q={data.location['lat']},{data.location['lng']}"
        
        admin_message_text = f"""🆕 Новый заказ №{order['order_number']}
👤 Пользователь: {user['full_name']} (@{user['username'] or 'нет'})
📞 Телефон: {data.phone}
💰 Сумма: {data.total:,.0f} сум
🕒 Время: {datetime.now().strftime('%H:%M %d.%m.%Y')}
📍 Локация: <a href="{location_url}">{location_text}</a>

🍽️ Заказ:
{items_text}

📝 Комментарий: {data.comment or 'нет'}"""
        
        admin_message = await self.bot.send_message(
            chat_id=config.SHOP_CHANNEL_ID,
            text=admin_message_text,
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="✅ Подтвержден", callback_data=f"confirm:{order['id']}"),
                    InlineKeyboardButton(text="🍳 Готовится", callback_data=f"cooking:{order['id']}")
                ],
                [
                    InlineKeyboardButton(text="🚴 Курьер", callback_data=f"assign_courier:{order['id']}")
                ],
                [
                    InlineKeyboardButton(text="❌ Отменить", callback_data=f"cancel:{order['id']}")
                ]
            ])
        )
        
        return order
    
    async def update_order_status(self, order_id: int, status: OrderStatus, courier_id: Optional[int] = None):
        await self.db.update_order_status(order_id, status, courier_id)
        
        # Get order details
        order = await self.db.get_order_by_id(order_id)
        
        status_texts = {
            OrderStatus.CONFIRMED: "✅ Подтвержден",
            OrderStatus.COOKING: "🍳 Готовится",
            OrderStatus.COURIER_ASSIGNED: "🚴 Курьер назначен",
            OrderStatus.OUT_FOR_DELIVERY: "📦 Передан курьеру",
            OrderStatus.DELIVERED: "🎉 Доставлен",
            OrderStatus.CANCELED: "❌ Отменен"
        }
        
        # Notify user
        if order and order['user_tg_id']:
            await self.bot.send_message(
                chat_id=order['user_tg_id'],
                text=f"📦 Заказ №{order['order_number']}\n"
                     f"Статус изменен: {status_texts.get(status, status.value)}"
            )
        
        # If courier assigned, notify courier
        if status == OrderStatus.COURIER_ASSIGNED and courier_id:
            courier = await self.db.get_courier_by_id(courier_id)
            if courier and order:
                location_url = f"https://maps.google.com/?q={order['location_lat']},{order['location_lng']}"
                await self.bot.send_message(
                    chat_id=courier['chat_id'],
                    text=f"""🚴 Новый заказ №{order['order_number']}
👤 Клиент: {order['customer_name']}
📞 Телефон: {order['phone']}
💰 Сумма: {order['total']:,.0f} сум
📍 Локация: <a href="{location_url}">На карте</a>

📝 Комментарий: {order['comment'] or 'нет'}""",
                    parse_mode=ParseMode.HTML,
                    reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                        [
                            InlineKeyboardButton(text="✅ Qabul qildim", callback_data=f"courier_accept:{order_id}"),
                            InlineKeyboardButton(text="📦 Yetkazildi", callback_data=f"courier_delivered:{order_id}")
                        ]
                    ])
                )

# ============================
# FastAPI WebApp Backend
# ============================

class FastAPIApp:
    def __init__(self, db: Database, bot: Bot):
        self.db = db
        self.bot = bot
        self.app = FastAPI(title="Telegram Food Delivery API", docs_url="/api/docs", redoc_url="/api/redoc")
        self.setup_middleware()
        self.setup_routes()
    
    def setup_middleware(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def verify_telegram_init_data(self, init_data: str) -> bool:
        """Verify Telegram WebApp initData"""
        try:
            # Parse initData
            data_pairs = init_data.split('&')
            data_dict = {}
            hash_value = None
            
            for pair in data_pairs:
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    if key == 'hash':
                        hash_value = value
                    else:
                        data_dict[key] = value
            
            if not hash_value:
                return False
            
            # Create data check string
            check_string = '\n'.join(
                f"{key}={data_dict[key]}"
                for key in sorted(data_dict.keys())
            )
            
            # Calculate secret key
            secret_key = hmac.new(
                key=b"WebAppData",
                msg=config.BOT_TOKEN.encode(),
                digestmod=hashlib.sha256
            ).digest()
            
            # Calculate hash
            calculated_hash = hmac.new(
                key=secret_key,
                msg=check_string.encode(),
                digestmod=hashlib.sha256
            ).hexdigest()
            
            return calculated_hash == hash_value
        except Exception as e:
            logging.error(f"Error verifying init data: {e}")
            return False
    
    def setup_routes(self):
        @self.app.get("/")
        async def root():
            return {"status": "ok", "service": "Telegram Food Delivery API"}
        
        @self.app.get("/api/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        @self.app.get("/api/foods")
        async def get_foods(initData: str):
            if not self.verify_telegram_init_data(initData):
                raise HTTPException(status_code=401, detail="Invalid initData")
            
            foods = await self.db.get_foods()
            return JSONResponse(content=[
                {
                    "id": f["id"],
                    "name": f["name"],
                    "description": f["description"],
                    "price": float(f["price"]),
                    "rating": float(f["rating"]),
                    "is_new": f["is_new"],
                    "category_id": f["category_id"],
                    "category_name": f.get("category_name", ""),
                    "image_url": f["image_url"]
                }
                for f in foods
            ])
        
        @self.app.get("/api/categories")
        async def get_categories(initData: str):
            if not self.verify_telegram_init_data(initData):
                raise HTTPException(status_code=401, detail="Invalid initData")
            
            categories = await self.db.get_categories()
            return JSONResponse(content=[
                {
                    "id": c["id"],
                    "name": c["name"]
                }
                for c in categories
            ])
        
        @self.app.get("/api/promo/validate")
        async def validate_promo(code: str, initData: str):
            if not self.verify_telegram_init_data(initData):
                raise HTTPException(status_code=401, detail="Invalid initData")
            
            promo = await self.db.validate_promo(code)
            if promo:
                return JSONResponse(content={
                    "valid": True,
                    "discount_percent": promo["discount_percent"],
                    "code": promo["code"],
                    "id": promo["id"]
                })
            return JSONResponse(content={"valid": False})
        
        @self.app.get("/webapp")
        async def webapp_index():
            # Return your Netlify hosted webapp URL
            return HTMLResponse(content=f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta http-equiv="refresh" content="0; url={config.WEBAPP_URL}">
            </head>
            <body>
                <p>Redirecting to <a href="{config.WEBAPP_URL}">Food Delivery WebApp</a>...</p>
            </body>
            </html>
            """)

# ============================
# Telegram Bot Handlers
# ============================

async def start_handler(message: Message, db: Database, bot: Bot):
    args = message.text.split()
    ref_by = None
    
    if len(args) > 1:
        try:
            ref_by = int(args[1])
        except:
            pass
    
    user = await db.create_user(
        tg_id=message.from_user.id,
        username=message.from_user.username,
        full_name=message.from_user.full_name,
        ref_by=ref_by
    )
    
    # Create main keyboard
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(text="🛍 Заказать", web_app=WebAppInfo(url=config.WEBAPP_URL)),
                KeyboardButton(text="📦 Мои заказы")
            ],
            [
                KeyboardButton(text="ℹ️ Информация о нас"),
                KeyboardButton(text="👥 Пригласить друга")
            ]
        ],
        resize_keyboard=True
    )
    
    welcome_text = f"""Добро пожаловать в FIESTA! {html.quote(message.from_user.full_name)}

Для заказа перейдите по кнопке ➡️
🛍 Заказать"""
    
    await message.answer(welcome_text, reply_markup=keyboard, parse_mode=ParseMode.HTML)

async def my_orders_handler(message: Message, db: Database):
    user = await db.get_user(message.from_user.id)
    if not user:
        await message.answer("Сначала зарегистрируйтесь через /start")
        return
    
    orders = await db.get_user_orders(user['id'])
    
    if not orders:
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🛍 Заказать", web_app=WebAppInfo(url=config.WEBAPP_URL))]
        ])
        await message.answer(
            "В данный момент у вас нет активных заказов в нашем магазине.\n"
            "Чтобы открыть магазин, нажмите кнопку ниже",
            reply_markup=keyboard
        )
        return
    
    text = "📦 Ваши заказы:\n\n"
    for order in orders[:10]:
        created_at = order['created_at'].strftime('%d.%m.%Y %H:%M')
        status_text = {
            'NEW': '🆕 Принят',
            'CONFIRMED': '✅ Подтвержден',
            'COOKING': '🍳 Готовится',
            'COURIER_ASSIGNED': '🚴 Курьер назначен',
            'OUT_FOR_DELIVERY': '📦 Передан курьеру',
            'DELIVERED': '🎉 Доставлен',
            'CANCELED': '❌ Отменен'
        }.get(order['status'], order['status'])
        
        text += f"{status_text} №{order['order_number']}\n"
        text += f"📅 {created_at} | 💰 {order['total']:,.0f} сум\n\n"
    
    await message.answer(text)

async def info_handler(message: Message):
    info_text = """🌟 Добро Пожаловать в FIESTA !

📍 Наш адрес: Хорезмская область, г.Хива, махаллинский сход граждан Гиламчи
🏢 Ориентир: Школа №12 Оруджева
📞 Контактный номер: +998 91 420 15 15
🕙 Рабочие часы: 24/7
📷 Мы в Instagram: fiesta.khiva (https://www.instagram.com/fiesta.khiva?igsh=Z3VoMzE0eGx0ZTVo)
🔗 Найти нас на карте: Место расположение (https://maps.app.goo.gl/dpBVHBWX1K7NTYVR7)"""
    
    await message.answer(info_text, disable_web_page_preview=False)

async def referral_handler(message: Message, db: Database, bot: Bot):
    user = await db.get_user(message.from_user.id)
    if not user:
        await message.answer("Сначала зарегистрируйтесь через /start")
        return
    
    stats = await db.get_referral_stats(user['id'])
    bot_username = (await bot.get_me()).username
    ref_link = f"https://t.me/{bot_username}?start={user['id']}"
    
    text = f"""За приглашение друга, вы можете получить промо-код от нас

👥 Вы пригласили {stats['ref_count']} человек
🛒 Оформили заказов: {stats['orders_count']}
💰 Оплатили заказов: {stats['delivered_count']}

👤 Ваша реферальная ссылка: {ref_link}

Пригласите трех человек и вы получите от нас промо-код со скидкой 15%"""
    
    # Check if user qualifies for promo
    if stats['ref_count'] >= 3:
        async with db.pool.acquire() as conn:
            existing_promo = await conn.fetchrow(
                "SELECT * FROM promos WHERE code LIKE $1",
                f"REF{user['id']}%"
            )
            
            if not existing_promo:
                # Create promo for user
                promo_code = f"REF{user['id']}{uuid.uuid4().hex[:4].upper()}"
                await conn.execute('''
                    INSERT INTO promos (code, discount_percent, usage_limit)
                    VALUES ($1, $2, $3)
                ''', promo_code, 15, 1)
                
                text += f"\n\n🎉 Поздравляем! Вы получили промо-код: {promo_code}"
    
    await message.answer(text)

async def web_app_data_handler(message: WebAppData, db: Database, bot: Bot):
    try:
        data = json.loads(message.web_app_data.data)
        
        if data.get('type') == 'order_create':
            order_data = WebAppOrderData(**data)
            
            user = await db.get_user(message.from_user.id)
            if not user:
                user = await db.create_user(
                    tg_id=message.from_user.id,
                    username=message.from_user.username,
                    full_name=message.from_user.full_name
                )
            
            order_service = OrderService(db, bot)
            await order_service.create_order_from_webapp(order_data, user['id'])
            
            await message.answer("✅ Ваш заказ успешно создан! Ожидайте подтверждения.")
    except Exception as e:
        logging.error(f"Error processing web app data: {e}")
        await message.answer("❌ Ошибка при обработке заказа. Попробуйте снова.")

async def admin_handler(message: Message):
    if message.from_user.id not in config.ADMIN_IDS:
        await message.answer("Доступ запрещен")
        return
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🍔 Taomlar", callback_data="admin_foods")],
        [InlineKeyboardButton(text="📂 Kategoriyalar", callback_data="admin_categories")],
        [InlineKeyboardButton(text="🎁 Promokodlar", callback_data="admin_promos")],
        [InlineKeyboardButton(text="📊 Statistika", callback_data="admin_stats")],
        [InlineKeyboardButton(text="🚴 Kuryerlar", callback_data="admin_couriers")],
        [InlineKeyboardButton(text="📦 Aktiv buyurtmalar", callback_data="admin_active_orders")],
        [InlineKeyboardButton(text="⚙️ Sozlamalar", callback_data="admin_settings")]
    ])
    
    await message.answer("👨‍💼 Админ панель:", reply_markup=keyboard)

async def admin_callback_handler(callback: CallbackQuery, db: Database, bot: Bot):
    if callback.from_user.id not in config.ADMIN_IDS:
        await callback.answer("Доступ запрещен")
        return
    
    data = callback.data
    
    if data.startswith("confirm:"):
        order_id = int(data.split(":")[1])
        order_service = OrderService(db, bot)
        await order_service.update_order_status(order_id, OrderStatus.CONFIRMED)
        await callback.answer("Заказ подтвержден")
        
        # Update message buttons
        await callback.message.edit_reply_markup(
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="✅ ПОДТВЕРЖДЕН", callback_data="noop"),
                    InlineKeyboardButton(text="🍳 Готовится", callback_data=f"cooking:{order_id}")
                ],
                [
                    InlineKeyboardButton(text="🚴 Курьер", callback_data=f"assign_courier:{order_id}")
                ]
            ])
        )
    
    elif data.startswith("cooking:"):
        order_id = int(data.split(":")[1])
        order_service = OrderService(db, bot)
        await order_service.update_order_status(order_id, OrderStatus.COOKING)
        await callback.answer("Заказ готовится")
        
        await callback.message.edit_reply_markup(
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="🍳 ГОТОВИТСЯ", callback_data="noop"),
                    InlineKeyboardButton(text="🚴 Курьер", callback_data=f"assign_courier:{order_id}")
                ]
            ])
        )
    
    elif data.startswith("cancel:"):
        order_id = int(data.split(":")[1])
        order_service = OrderService(db, bot)
        await order_service.update_order_status(order_id, OrderStatus.CANCELED)
        await callback.answer("Заказ отменен")
        
        await callback.message.edit_reply_markup(
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="❌ ОТМЕНЕН", callback_data="noop")
                ]
            ])
        )
    
    elif data.startswith("assign_courier:"):
        order_id = int(data.split(":")[1])
        couriers = await db.get_couriers()
        
        if not couriers:
            await callback.answer("Нет активных курьеров")
            return
        
        keyboard = InlineKeyboardBuilder()
        for courier in couriers:
            keyboard.button(
                text=f"🚴 {courier['name']}",
                callback_data=f"assign_courier_to:{order_id}:{courier['id']}"
            )
        keyboard.adjust(1)
        
        await callback.message.answer(
            f"Выберите курьера для заказа №{order_id}",
            reply_markup=keyboard.as_markup()
        )
        await callback.answer()
    
    elif data.startswith("assign_courier_to:"):
        _, order_id, courier_id = data.split(":")
        order_id = int(order_id)
        courier_id = int(courier_id)
        
        order_service = OrderService(db, bot)
        await order_service.update_order_status(order_id, OrderStatus.COURIER_ASSIGNED, courier_id)
        await callback.answer("Курьер назначен")
        
        await callback.message.edit_reply_markup(
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="🚴 КУРЬЕР НАЗНАЧЕН", callback_data="noop"),
                    InlineKeyboardButton(text="📦 Передан курьеру", callback_data=f"out_for_delivery:{order_id}")
                ]
            ])
        )
    
    elif data.startswith("out_for_delivery:"):
        order_id = int(data.split(":")[1])
        order_service = OrderService(db, bot)
        await order_service.update_order_status(order_id, OrderStatus.OUT_FOR_DELIVERY)
        await callback.answer("Заказ передан курьеру")
        
        await callback.message.edit_reply_markup(
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="📦 ПЕРЕДАН КУРЬЕРУ", callback_data="noop")
                ]
            ])
        )
    
    elif data.startswith("courier_accept:"):
        order_id = int(data.split(":")[1])
        order_service = OrderService(db, bot)
        await order_service.update_order_status(order_id, OrderStatus.OUT_FOR_DELIVERY)
        await callback.answer("Заказ принят")
        
        await callback.message.edit_text(
            callback.message.text + "\n\n✅ Курьер принял заказ",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="📦 Yetkazildi", callback_data=f"courier_delivered:{order_id}")]
            ])
        )
    
    elif data.startswith("courier_delivered:"):
        order_id = int(data.split(":")[1])
        order_service = OrderService(db, bot)
        await order_service.update_order_status(order_id, OrderStatus.DELIVERED)
        await callback.answer("Заказ доставлен")
        
        await callback.message.edit_text(
            callback.message.text + "\n\n🎉 Заказ доставлен!",
            reply_markup=None
        )
    
    elif data == "admin_active_orders":
        orders = await db.get_active_orders()
        
        if not orders:
            await callback.message.answer("Нет активных заказов")
        else:
            text = "📦 Активные заказы:\n\n"
            for order in orders[:20]:
                created_at = order['created_at'].strftime('%H:%M %d.%m')
                status_text = {
                    'NEW': '🆕 Принят',
                    'CONFIRMED': '✅ Подтвержден',
                    'COOKING': '🍳 Готовится',
                    'COURIER_ASSIGNED': '🚴 Курьер назначен',
                    'OUT_FOR_DELIVERY': '📦 В пути'
                }.get(order['status'], order['status'])
                
                text += f"{status_text} №{order['order_number']}\n"
                text += f"👤 {order['customer_name']} | 📞 {order['phone']}\n"
                text += f"📅 {created_at} | 💰 {order['total']:,.0f} сум\n\n"
            
            await callback.message.answer(text)
        await callback.answer()
    
    elif data == "admin_stats":
        # Today stats
        today_stats = await db.get_stats(1)
        week_stats = await db.get_stats(7)
        month_stats = await db.get_stats(30)
        
        text = f"""📊 Статистика:

📅 Сегодня:
├ Заказы: {today_stats['orders_count']}
├ Доставлено: {today_stats['delivered_count']}
└ Выручка: {today_stats['revenue']:,.0f} сум

📅 За неделю:
├ Заказы: {week_stats['orders_count']}
├ Доставлено: {week_stats['delivered_count']}
└ Выручка: {week_stats['revenue']:,.0f} сум

📅 За месяц:
├ Заказы: {month_stats['orders_count']}
├ Доставлено: {month_stats['delivered_count']}
└ Выручка: {month_stats['revenue']:,.0f} сум"""
        
        await callback.message.answer(text)
        await callback.answer()
    
    elif data == "admin_couriers":
        couriers = await db.get_couriers(active_only=False)
        
        if not couriers:
            text = "Нет зарегистрированных курьеров"
        else:
            text = "🚴 Список курьеров:\n\n"
            for courier in couriers:
                status = "✅ Активен" if courier['is_active'] else "❌ Неактивен"
                text += f"{courier['name']} - {status}\n"
                text += f"ID: {courier['chat_id']}\n\n"
        
        await callback.message.answer(text)
        await callback.answer()

# ============================
# Main Application
# ============================

async def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("bot.log")
        ]
    )
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting Telegram Food Delivery Bot...")
        
        # Initialize bot
        bot = Bot(token=config.BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
        
        # Initialize Redis storage
        redis_client = redis.from_url(config.REDIS_URL)
        storage = RedisStorage(redis=redis_client)
        
        # Initialize dispatcher
        dp = Dispatcher(storage=storage)
        
        # Initialize database
        logger.info("Connecting to database...")
        db = Database(config.DB_URL)
        await db.connect()
        logger.info("Database connected successfully")
        
        # Get bot username
        bot_info = await bot.get_me()
        config.BOT_USERNAME = bot_info.username
        logger.info(f"Bot @{config.BOT_USERNAME} is starting...")
        
        # Register handlers
        @dp.message(CommandStart())
        async def cmd_start(message: Message):
            await start_handler(message, db, bot)
        
        @dp.message(F.text == "📦 Мои заказы")
        async def cmd_my_orders(message: Message):
            await my_orders_handler(message, db)
        
        @dp.message(F.text == "ℹ️ Информация о нас")
        async def cmd_info(message: Message):
            await info_handler(message)
        
        @dp.message(F.text == "👥 Пригласить друга")
        async def cmd_referral(message: Message):
            await referral_handler(message, db, bot)
        
        @dp.message(Command("admin"))
        async def cmd_admin(message: Message):
            await admin_handler(message)
        
        @dp.message(F.web_app_data)
        async def handle_web_app_data(message: WebAppData):
            await web_app_data_handler(message, db, bot)
        
        @dp.callback_query()
        async def handle_callback(callback: CallbackQuery):
            await admin_callback_handler(callback, db, bot)
        
        # Initialize FastAPI
        fastapi_app = FastAPIApp(db, bot)
        
        # Start FastAPI server in background
        import threading
        
        def run_fastapi():
            uvicorn.run(
                fastapi_app.app,
                host="0.0.0.0",
                port=config.PORT,
                log_level="info",
                access_log=True
            )
        
        fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
        fastapi_thread.start()
        logger.info(f"FastAPI server started on port {config.PORT}")
        
        # Delete webhook and start polling
        await bot.delete_webhook(drop_pending_updates=True)
        logger.info("Webhook deleted, starting polling...")
        
        # Start bot
        await dp.start_polling(bot)
        
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
