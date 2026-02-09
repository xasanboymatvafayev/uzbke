"""
FIESTA Food Delivery Bot
Full-stack Telegram bot with WebApp, Admin Panel, and Courier System
"""

import asyncio
import logging
import os
import json
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from urllib.parse import parse_qsl

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.redis import RedisStorage
from aiogram.types import (
    Message, CallbackQuery, ReplyKeyboardMarkup, KeyboardButton,
    InlineKeyboardMarkup, InlineKeyboardButton, WebAppInfo
)

from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Integer, Float, Boolean, DateTime, Text, BigInteger, ForeignKey

from redis.asyncio import Redis

# ==================== CONFIGURATION ====================
BOT_TOKEN = os.getenv("BOT_TOKEN", "7917271389:AAE4PXCowGo6Bsfdy3Hrz3x689MLJdQmVi4")
ADMIN_IDS = [int(x.strip()) for x in os.getenv("ADMIN_IDS", "6365371142").split(",")]
DB_URL = os.getenv("DB_URL", "postgresql+asyncpg://postgres:BDAaILJKOITNLlMOjJNfWiRPbICwEcpZ@centerbeam.proxy.rlwy.net:35489/railway")
REDIS_URL = os.getenv("REDIS_URL", "redis://default:GBrZNeUKJfqRlPcQUoUICWQpbQRtRRJp@ballast.proxy.rlwy.net:35411")
SHOP_CHANNEL_ID = os.getenv("SHOP_CHANNEL_ID", "-1003530497437")
COURIER_CHANNEL_ID = os.getenv("COURIER_CHANNEL_ID", "-1003707946746")
WEBAPP_URL = os.getenv("WEBAPP_URL", "https://mainsufooduz.netlify.app")
BACKEND_URL = os.getenv("BACKEND_URL", "https://uzbke-production.up.railway.app")

# ==================== DATABASE MODELS ====================
class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    tg_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    username: Mapped[Optional[str]] = mapped_column(String(255))
    full_name: Mapped[str] = mapped_column(String(255))
    joined_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    ref_by_user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"))
    promo_rewarded: Mapped[bool] = mapped_column(Boolean, default=False)


class Category(Base):
    __tablename__ = "categories"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class Food(Base):
    __tablename__ = "foods"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    category_id: Mapped[int] = mapped_column(ForeignKey("categories.id"))
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[Optional[str]] = mapped_column(Text)
    price: Mapped[float] = mapped_column(Float)
    rating: Mapped[float] = mapped_column(Float, default=5.0)
    is_new: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    image_url: Mapped[Optional[str]] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Order(Base):
    __tablename__ = "orders"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    order_number: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    customer_name: Mapped[str] = mapped_column(String(255))
    phone: Mapped[str] = mapped_column(String(20))
    comment: Mapped[Optional[str]] = mapped_column(Text)
    total: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(50), default="NEW")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    delivered_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    location_lat: Mapped[float] = mapped_column(Float)
    location_lng: Mapped[float] = mapped_column(Float)
    courier_id: Mapped[Optional[int]] = mapped_column(ForeignKey("couriers.id"))
    admin_message_id: Mapped[Optional[int]] = mapped_column(Integer)


class OrderItem(Base):
    __tablename__ = "order_items"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"))
    food_id: Mapped[int] = mapped_column(ForeignKey("foods.id"))
    name_snapshot: Mapped[str] = mapped_column(String(255))
    price_snapshot: Mapped[float] = mapped_column(Float)
    qty: Mapped[int] = mapped_column(Integer)
    line_total: Mapped[float] = mapped_column(Float)


class Promo(Base):
    __tablename__ = "promos"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(50), unique=True)
    discount_percent: Mapped[int] = mapped_column(Integer)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    usage_limit: Mapped[Optional[int]] = mapped_column(Integer)
    used_count: Mapped[int] = mapped_column(Integer, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class Courier(Base):
    __tablename__ = "couriers"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, unique=True)
    name: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# ==================== DATABASE SESSION ====================
engine = create_async_engine(DB_URL, echo=False)
async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Insert demo data
    async with async_session_maker() as session:
        # Check if categories exist
        result = await session.execute(select(Category))
        if not result.scalars().first():
            categories = [
                Category(name="Lavash"), Category(name="Burger"), Category(name="Xaggi"),
                Category(name="Shaurma"), Category(name="Hotdog"), Category(name="Combo"),
                Category(name="Sneki"), Category(name="Sous"), Category(name="Napitki")
            ]
            session.add_all(categories)
            await session.commit()
            
            # Add demo foods
            foods = [
                # Lavash
                Food(category_id=1, name="Lavash Mini", description="Kichik lavash", price=15000, rating=4.8, is_new=True),
                Food(category_id=1, name="Lavash Standart", description="O'rtacha lavash", price=22000, rating=4.9),
                Food(category_id=1, name="Lavash Maksi", description="Katta lavash", price=28000, rating=5.0),
                # Burger
                Food(category_id=2, name="Gamburger", description="Klassik burger", price=20000, rating=4.7),
                Food(category_id=2, name="Chizburger", description="Pishloqli burger", price=25000, rating=4.9),
                Food(category_id=2, name="Double Burger", description="Ikki qatlam", price=35000, rating=5.0, is_new=True),
                # Xaggi
                Food(category_id=3, name="Xaggi Mini", description="Kichik xaggi", price=12000, rating=4.5),
                Food(category_id=3, name="Xaggi Standart", description="O'rtacha xaggi", price=18000, rating=4.8),
                Food(category_id=3, name="Xaggi Maksi", description="Katta xaggi", price=24000, rating=4.9),
                # Shaurma
                Food(category_id=4, name="Shaurma Tovuq", description="Tovuq go'shtli", price=20000, rating=4.8),
                Food(category_id=4, name="Shaurma Mol", description="Mol go'shtli", price=25000, rating=4.9),
                Food(category_id=4, name="Shaurma Mix", description="Aralash", price=28000, rating=5.0),
                # Hotdog
                Food(category_id=5, name="Hotdog Mini", description="Kichik hotdog", price=10000, rating=4.5),
                Food(category_id=5, name="Hotdog Standart", description="O'rtacha hotdog", price=15000, rating=4.7),
                Food(category_id=5, name="Hotdog Maksi", description="Katta hotdog", price=20000, rating=4.8),
                # Combo
                Food(category_id=6, name="Combo 1", description="Lavash + Pepsi", price=30000, rating=4.9),
                Food(category_id=6, name="Combo 2", description="Burger + Fri + Cola", price=40000, rating=5.0, is_new=True),
                Food(category_id=6, name="Combo 3", description="Shaurma + Napitka", price=35000, rating=4.8),
                # Sneki
                Food(category_id=7, name="Fri Mini", description="Kichik kartoshka fri", price=8000, rating=4.6),
                Food(category_id=7, name="Fri Standart", description="O'rtacha kartoshka fri", price=12000, rating=4.8),
                Food(category_id=7, name="Nagets 6 dona", description="Tovuq nagetslari", price=15000, rating=4.7),
                # Sous
                Food(category_id=8, name="Ketchup", description="Pomidor sousi", price=2000, rating=4.5),
                Food(category_id=8, name="Mayonez", description="Mayonez sousi", price=2000, rating=4.5),
                Food(category_id=8, name="Chili", description="O'tkir sous", price=3000, rating=4.7),
                # Napitki
                Food(category_id=9, name="Pepsi 0.5L", description="Gazlangan ichimlik", price=7000, rating=4.6),
                Food(category_id=9, name="Coca Cola 0.5L", description="Gazlangan ichimlik", price=7000, rating=4.6),
                Food(category_id=9, name="Fanta 0.5L", description="Gazlangan ichimlik", price=7000, rating=4.5),
            ]
            session.add_all(foods)
            await session.commit()


# ==================== FSM STATES ====================
class AdminStates(StatesGroup):
    # Food management
    add_food_name = State()
    add_food_category = State()
    add_food_price = State()
    add_food_description = State()
    add_food_rating = State()
    add_food_image = State()
    
    # Category management
    add_category_name = State()
    
    # Promo management
    add_promo_code = State()
    add_promo_discount = State()
    add_promo_expires = State()
    add_promo_limit = State()
    
    # Courier management
    add_courier_chat_id = State()
    add_courier_name = State()


# ==================== KEYBOARDS ====================
def get_main_keyboard():
    """Main client keyboard"""
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🛍 Заказать", web_app=WebAppInfo(url=WEBAPP_URL))],
            [KeyboardButton(text="📦 Мои заказы"), KeyboardButton(text="ℹ️ Информация о нас")],
            [KeyboardButton(text="👥 Пригласить друга")]
        ],
        resize_keyboard=True
    )
    return keyboard


def get_admin_main_keyboard():
    """Admin panel main menu"""
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🍔 Таомлар", callback_data="admin_foods")],
        [InlineKeyboardButton(text="📂 Категориялар", callback_data="admin_categories")],
        [InlineKeyboardButton(text="🎁 Промокодлар", callback_data="admin_promos")],
        [InlineKeyboardButton(text="📊 Статистика", callback_data="admin_stats")],
        [InlineKeyboardButton(text="🚴 Курьерлар", callback_data="admin_couriers")],
        [InlineKeyboardButton(text="📦 Актив буюртмалар", callback_data="admin_active_orders")]
    ])
    return keyboard


def get_order_admin_keyboard(order_id: int, status: str):
    """Order management keyboard for admin channel"""
    buttons = []
    
    if status == "NEW":
        buttons.append([InlineKeyboardButton(text="✅ Подтверждён", callback_data=f"order_confirm:{order_id}")])
    
    if status in ["NEW", "CONFIRMED"]:
        buttons.append([InlineKeyboardButton(text="🍳 Готовится", callback_data=f"order_cooking:{order_id}")])
    
    if status in ["NEW", "CONFIRMED", "COOKING"]:
        buttons.append([InlineKeyboardButton(text="🚴 Назначить курьера", callback_data=f"order_assign_courier:{order_id}")])
    
    buttons.append([InlineKeyboardButton(text="❌ Отменить", callback_data=f"order_cancel:{order_id}")])
    
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def get_courier_select_keyboard(order_id: int, couriers: List[Courier]):
    """Courier selection keyboard"""
    buttons = []
    for courier in couriers:
        buttons.append([InlineKeyboardButton(
            text=f"🚴 {courier.name}",
            callback_data=f"assign_courier:{order_id}:{courier.id}"
        )])
    buttons.append([InlineKeyboardButton(text="◀️ Назад", callback_data="admin_active_orders")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def get_courier_order_keyboard(order_id: int):
    """Courier order management keyboard"""
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="✅ Қабул қилдим", callback_data=f"courier_accept:{order_id}")],
        [InlineKeyboardButton(text="📦 Етказилди", callback_data=f"courier_delivered:{order_id}")]
    ])
    return keyboard


# ==================== HELPER FUNCTIONS ====================
async def get_user_by_tg_id(session: AsyncSession, tg_id: int) -> Optional[User]:
    """Get user by telegram ID"""
    result = await session.execute(select(User).where(User.tg_id == tg_id))
    return result.scalar_one_or_none()


async def get_or_create_user(session: AsyncSession, tg_id: int, username: str, full_name: str, ref_id: Optional[int] = None) -> User:
    """Get existing user or create new one"""
    user = await get_user_by_tg_id(session, tg_id)
    
    if not user:
        # Check referral
        ref_user = None
        if ref_id and ref_id != tg_id:
            ref_user = await session.execute(select(User).where(User.tg_id == ref_id))
            ref_user = ref_user.scalar_one_or_none()
        
        user = User(
            tg_id=tg_id,
            username=username,
            full_name=full_name,
            ref_by_user_id=ref_user.id if ref_user else None
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
    
    return user


async def get_referral_stats(session: AsyncSession, user_id: int) -> Dict[str, int]:
    """Get referral statistics for user"""
    # Count referred users
    ref_count_result = await session.execute(
        select(func.count(User.id)).where(User.ref_by_user_id == user_id)
    )
    ref_count = ref_count_result.scalar() or 0
    
    # Count orders from referred users
    referred_users = await session.execute(select(User.id).where(User.ref_by_user_id == user_id))
    referred_user_ids = [row[0] for row in referred_users.fetchall()]
    
    orders_count = 0
    paid_count = 0
    
    if referred_user_ids:
        orders_result = await session.execute(
            select(func.count(Order.id)).where(Order.user_id.in_(referred_user_ids))
        )
        orders_count = orders_result.scalar() or 0
        
        paid_result = await session.execute(
            select(func.count(Order.id)).where(
                and_(Order.user_id.in_(referred_user_ids), Order.status == "DELIVERED")
            )
        )
        paid_count = paid_result.scalar() or 0
    
    return {
        "ref_count": ref_count,
        "orders_count": orders_count,
        "paid_count": paid_count
    }


async def create_referral_promo(session: AsyncSession, user: User) -> Promo:
    """Create 15% promo code for user"""
    code = f"REF15_{user.tg_id}"
    expires_at = datetime.utcnow() + timedelta(days=30)
    
    promo = Promo(
        code=code,
        discount_percent=15,
        expires_at=expires_at,
        usage_limit=1,
        is_active=True
    )
    session.add(promo)
    
    user.promo_rewarded = True
    await session.commit()
    await session.refresh(promo)
    
    return promo


async def send_order_to_admin_channel(bot: Bot, order: Order, user: User, items: List[OrderItem]):
    """Send new order to admin channel"""
    items_text = "\n".join([
        f"{item.name_snapshot} x{item.qty} = {item.line_total:,.0f} сум"
        for item in items
    ])
    
    location_link = f"https://maps.google.com/?q={order.location_lat},{order.location_lng}"
    
    text = f"""🆕 <b>Новый заказ №{order.order_number}</b>

👤 Пользователь: {user.full_name} (@{user.username or 'без username'})
📞 Телефон: {order.phone}
💰 Сумма: {order.total:,.0f} сум
🕒 Время: {order.created_at.strftime('%d.%m.%Y %H:%M')}
📍 <a href="{location_link}">Локация на карте</a>

🍽️ Заказ:
{items_text}

💬 Комментарий: {order.comment or 'Нет'}
"""
    
    keyboard = get_order_admin_keyboard(order.id, order.status)
    
    msg = await bot.send_message(
        SHOP_CHANNEL_ID,
        text,
        reply_markup=keyboard,
        parse_mode="HTML"
    )
    
    # Save message ID
    async with async_session_maker() as session:
        order_obj = await session.get(Order, order.id)
        order_obj.admin_message_id = msg.message_id
        await session.commit()


async def update_order_status(session: AsyncSession, bot: Bot, order_id: int, new_status: str, courier_id: Optional[int] = None):
    """Update order status and notify"""
    order = await session.get(Order, order_id)
    if not order:
        return
    
    old_status = order.status
    order.status = new_status
    order.updated_at = datetime.utcnow()
    
    if courier_id:
        order.courier_id = courier_id
    
    if new_status == "DELIVERED":
        order.delivered_at = datetime.utcnow()
    
    await session.commit()
    
    # Status translations
    status_names = {
        "NEW": "Принят",
        "CONFIRMED": "Подтверждён",
        "COOKING": "Готовится",
        "COURIER_ASSIGNED": "Курьер назначен",
        "OUT_FOR_DELIVERY": "Передан курьеру",
        "DELIVERED": "Доставлен",
        "CANCELED": "Отменён"
    }
    
    # Notify user
    user = await session.get(User, order.user_id)
    status_emoji = {
        "CONFIRMED": "✅",
        "COOKING": "🍳",
        "COURIER_ASSIGNED": "🚴",
        "OUT_FOR_DELIVERY": "🚴‍♂️",
        "DELIVERED": "🎉",
        "CANCELED": "❌"
    }
    
    emoji = status_emoji.get(new_status, "📦")
    
    await bot.send_message(
        user.tg_id,
        f"{emoji} <b>Ваш заказ №{order.order_number}</b>\n\n"
        f"Статус: {status_names.get(new_status, new_status)}",
        parse_mode="HTML"
    )
    
    # Update admin channel message
    if order.admin_message_id:
        try:
            items = await session.execute(select(OrderItem).where(OrderItem.order_id == order.id))
            items_list = items.scalars().all()
            items_text = "\n".join([
                f"{item.name_snapshot} x{item.qty} = {item.line_total:,.0f} сум"
                for item in items_list
            ])
            
            location_link = f"https://maps.google.com/?q={order.location_lat},{order.location_lng}"
            
            status_text = f"📦 <b>Статус: {status_names.get(new_status, new_status)}</b>"
            if new_status == "DELIVERED":
                status_text = f"✅ <b>ДОСТАВЛЕН</b> ({order.delivered_at.strftime('%d.%m.%Y %H:%M')})"
            
            text = f"""🆔 <b>Заказ №{order.order_number}</b>
{status_text}

👤 Пользователь: {user.full_name} (@{user.username or 'без username'})
📞 Телефон: {order.phone}
💰 Сумма: {order.total:,.0f} сум
🕒 Время: {order.created_at.strftime('%d.%m.%Y %H:%M')}
📍 <a href="{location_link}">Локация на карте</a>

🍽️ Заказ:
{items_text}

💬 Комментарий: {order.comment or 'Нет'}
"""
            
            # Remove keyboard if delivered or canceled
            keyboard = None if new_status in ["DELIVERED", "CANCELED"] else get_order_admin_keyboard(order.id, new_status)
            
            await bot.edit_message_text(
                text,
                SHOP_CHANNEL_ID,
                order.admin_message_id,
                reply_markup=keyboard,
                parse_mode="HTML"
            )
        except Exception as e:
            logging.error(f"Failed to update admin message: {e}")


async def send_order_to_courier(bot: Bot, order: Order, courier: Courier, user: User, items: List[OrderItem]):
    """Send order to courier"""
    items_text = "\n".join([
        f"{item.name_snapshot} x{item.qty}"
        for item in items
    ])
    
    location_link = f"https://maps.google.com/?q={order.location_lat},{order.location_lng}"
    
    text = f"""🚴 <b>Новый заказ №{order.order_number}</b>

👤 Клиент: {order.customer_name}
📞 Телефон: {order.phone}
💰 Сумма: {order.total:,.0f} сум
📍 <a href="{location_link}">Локация на карте</a>

🍽️ Список:
{items_text}

💬 Комментарий: {order.comment or 'Нет'}
"""
    
    keyboard = get_courier_order_keyboard(order.id)
    
    await bot.send_message(
        courier.chat_id,
        text,
        reply_markup=keyboard,
        parse_mode="HTML"
    )


# ==================== BOT HANDLERS ====================
router = Router()


@router.message(Command("start"))
async def cmd_start(message: Message):
    """Handle /start command"""
    args = message.text.split()
    ref_id = None
    
    if len(args) > 1:
        try:
            ref_id = int(args[1])
        except:
            pass
    
    async with async_session_maker() as session:
        user = await get_or_create_user(
            session,
            message.from_user.id,
            message.from_user.username or "",
            message.from_user.full_name,
            ref_id
        )
    
    text = f"""Добро пожаловать в FIESTA! {message.from_user.full_name}

Для заказа перейдите по кнопке ➡️
🛍 Заказать"""
    
    await message.answer(text, reply_markup=get_main_keyboard())


@router.message(F.text == "📦 Мои заказы")
async def my_orders(message: Message):
    """Show user's orders"""
    async with async_session_maker() as session:
        user = await get_user_by_tg_id(session, message.from_user.id)
        if not user:
            await message.answer("Пожалуйста, нажмите /start")
            return
        
        # Get last 10 orders
        result = await session.execute(
            select(Order)
            .where(Order.user_id == user.id)
            .order_by(desc(Order.created_at))
            .limit(10)
        )
        orders = result.scalars().all()
        
        if not orders:
            await message.answer(
                "В данный момент у вас нет активных заказов в нашем магазине.\n"
                "Чтобы открыть магазин, введите команду — /shop"
            )
            return
        
        status_names = {
            "NEW": "Принят",
            "CONFIRMED": "Подтверждён",
            "COOKING": "Готовится",
            "COURIER_ASSIGNED": "Курьер назначен",
            "OUT_FOR_DELIVERY": "Передан курьеру",
            "DELIVERED": "Доставлен",
            "CANCELED": "Отменён"
        }
        
        text = "📦 <b>Ваши заказы:</b>\n\n"
        
        for order in orders:
            items_result = await session.execute(
                select(OrderItem).where(OrderItem.order_id == order.id)
            )
            items = items_result.scalars().all()
            
            items_text = "\n".join([f"  • {item.name_snapshot} x{item.qty}" for item in items])
            
            text += f"""🆔 Заказ №{order.order_number}
📅 {order.created_at.strftime('%d.%m.%Y %H:%M')}
💰 {order.total:,.0f} сум
📦 {status_names.get(order.status, order.status)}

{items_text}

━━━━━━━━━━━━━━━━━━━━

"""
        
        await message.answer(text, parse_mode="HTML")


@router.message(F.text == "ℹ️ Информация о нас")
async def info_about_us(message: Message):
    """Show info about restaurant"""
    text = """🌟 Добро Пожаловать в FIESTA!

📍 Наш адрес: Хорезмская область, г.Хива, махаллинский сход граждан Гиламчи
🏢 Ориентир: Школа №12 Оруджева
📞 Контактный номер: +998 91 420 15 15
🕙 Рабочие часы: 24/7

📷 Мы в Instagram: <a href="https://www.instagram.com/fiesta.khiva">fiesta.khiva</a>
🔗 <a href="https://maps.app.goo.gl/dpBVHBWX1K7NTYVR7">Найти нас на карте</a>"""
    
    await message.answer(text, parse_mode="HTML", disable_web_page_preview=True)


@router.message(F.text == "👥 Пригласить друга")
async def invite_friend(message: Message):
    """Show referral info"""
    async with async_session_maker() as session:
        user = await get_user_by_tg_id(session, message.from_user.id)
        if not user:
            await message.answer("Пожалуйста, нажмите /start")
            return
        
        stats = await get_referral_stats(session, user.id)
        
        bot_me = await message.bot.get_me()
        ref_link = f"https://t.me/{bot_me.username}?start={user.tg_id}"
        
        text = f"""За приглашение друга, вы можете получить промо-код от нас

👥 Вы пригласили {stats['ref_count']} человек
🛒 Оформили заказов: {stats['orders_count']}
💰 Оплатили заказов: {stats['paid_count']}

👤 Ваша реферальная ссылка:
{ref_link}

Пригласите трех человек и вы получите от нас промо-код со скидкой 15%"""
        
        # Check if user should get promo
        if stats['ref_count'] >= 3 and not user.promo_rewarded:
            promo = await create_referral_promo(session, user)
            text += f"\n\n🎁 <b>Поздравляем! Ваш промокод: {promo.code}</b>"
        
        await message.answer(text, parse_mode="HTML")


@router.message(Command("shop"))
async def cmd_shop(message: Message):
    """Open shop via WebApp"""
    text = "Чтобы открыть наш магазин, нажмите кнопку ниже"
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🛍 Заказать", web_app=WebAppInfo(url=WEBAPP_URL))]
    ])
    await message.answer(text, reply_markup=keyboard)


@router.message(F.web_app_data)
async def handle_webapp_data(message: Message):
    """Handle order from WebApp"""
    try:
        data = json.loads(message.web_app_data.data)
        
        if data.get("type") != "order_create":
            await message.answer("❌ Неверный формат данных")
            return
        
        # Validate
        if data.get("total", 0) < 50000:
            await message.answer("❌ Минимальная сумма заказа 50,000 сум")
            return
        
        async with async_session_maker() as session:
            user = await get_user_by_tg_id(session, message.from_user.id)
            if not user:
                await message.answer("❌ Пожалуйста, нажмите /start")
                return
            
            # Generate order number
            last_order = await session.execute(
                select(Order).order_by(desc(Order.id)).limit(1)
            )
            last = last_order.scalar_one_or_none()
            order_number = f"ORD{(last.id + 1) if last else 1:05d}"
            
            # Create order
            order = Order(
                order_number=order_number,
                user_id=user.id,
                customer_name=data.get("customer_name", user.full_name),
                phone=data.get("phone", ""),
                comment=data.get("comment", ""),
                total=data.get("total", 0),
                status="NEW",
                location_lat=data.get("location", {}).get("lat", 0),
                location_lng=data.get("location", {}).get("lng", 0)
            )
            session.add(order)
            await session.flush()
            
            # Create order items
            items = []
            for item_data in data.get("items", []):
                item = OrderItem(
                    order_id=order.id,
                    food_id=item_data.get("food_id", 0),
                    name_snapshot=item_data.get("name", ""),
                    price_snapshot=item_data.get("price", 0),
                    qty=item_data.get("qty", 1),
                    line_total=item_data.get("price", 0) * item_data.get("qty", 1)
                )
                items.append(item)
                session.add(item)
            
            await session.commit()
            await session.refresh(order)
            
            # Notify user
            await message.answer(
                f"Ваш заказ принят ✅\n\n"
                f"🆔 Заказ №{order.order_number}\n"
                f"💰 Сумма: {order.total:,.0f} сум\n"
                f"📦 Статус: Принят"
            )
            
            # Send to admin channel
            await send_order_to_admin_channel(message.bot, order, user, items)
            
    except Exception as e:
        logging.error(f"Error handling webapp data: {e}")
        await message.answer("❌ Ошибка при обработке заказа. Попробуйте снова.")


# ==================== ADMIN HANDLERS ====================
@router.message(Command("admin"))
async def cmd_admin(message: Message):
    """Admin panel"""
    if message.from_user.id not in ADMIN_IDS:
        await message.answer("❌ У вас нет доступа к админ панели")
        return
    
    await message.answer("🔧 <b>Админ панель FIESTA</b>", reply_markup=get_admin_main_keyboard(), parse_mode="HTML")


@router.callback_query(F.data == "admin_stats")
async def admin_stats(callback: CallbackQuery):
    """Show statistics"""
    if callback.from_user.id not in ADMIN_IDS:
        await callback.answer("❌ Нет доступа", show_alert=True)
        return
    
    async with async_session_maker() as session:
        # Today stats
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        today_orders = await session.execute(
            select(func.count(Order.id)).where(Order.created_at >= today)
        )
        today_orders_count = today_orders.scalar() or 0
        
        today_delivered = await session.execute(
            select(func.count(Order.id)).where(
                and_(Order.created_at >= today, Order.status == "DELIVERED")
            )
        )
        today_delivered_count = today_delivered.scalar() or 0
        
        today_revenue = await session.execute(
            select(func.sum(Order.total)).where(
                and_(Order.created_at >= today, Order.status == "DELIVERED")
            )
        )
        today_revenue_sum = today_revenue.scalar() or 0
        
        # Active orders
        active_orders = await session.execute(
            select(func.count(Order.id)).where(
                Order.status.in_(["NEW", "CONFIRMED", "COOKING", "COURIER_ASSIGNED", "OUT_FOR_DELIVERY"])
            )
        )
        active_count = active_orders.scalar() or 0
        
        text = f"""📊 <b>Статистика</b>

<b>Сегодня:</b>
📦 Заказов: {today_orders_count}
✅ Доставлено: {today_delivered_count}
💰 Выручка: {today_revenue_sum:,.0f} сум

<b>Текущие:</b>
📦 Активных заказов: {active_count}
"""
    
    await callback.message.edit_text(text, reply_markup=get_admin_main_keyboard(), parse_mode="HTML")
    await callback.answer()


@router.callback_query(F.data == "admin_active_orders")
async def admin_active_orders(callback: CallbackQuery):
    """Show active orders"""
    if callback.from_user.id not in ADMIN_IDS:
        await callback.answer("❌ Нет доступа", show_alert=True)
        return
    
    async with async_session_maker() as session:
        result = await session.execute(
            select(Order).where(
                Order.status.in_(["NEW", "CONFIRMED", "COOKING", "COURIER_ASSIGNED", "OUT_FOR_DELIVERY"])
            ).order_by(desc(Order.created_at))
        )
        orders = result.scalars().all()
        
        if not orders:
            await callback.answer("Нет активных заказов", show_alert=True)
            return
        
        status_names = {
            "NEW": "Принят",
            "CONFIRMED": "Подтверждён",
            "COOKING": "Готовится",
            "COURIER_ASSIGNED": "Курьер назначен",
            "OUT_FOR_DELIVERY": "Передан курьеру"
        }
        
        text = "📦 <b>Активные заказы:</b>\n\n"
        
        for order in orders[:10]:
            text += f"🆔 №{order.order_number} | {status_names.get(order.status)}\n"
            text += f"💰 {order.total:,.0f} сум | 🕒 {order.created_at.strftime('%H:%M')}\n\n"
        
        await callback.message.edit_text(text, reply_markup=get_admin_main_keyboard(), parse_mode="HTML")
    
    await callback.answer()


@router.callback_query(F.data.startswith("order_confirm:"))
async def order_confirm(callback: CallbackQuery):
    """Confirm order"""
    if callback.from_user.id not in ADMIN_IDS:
        await callback.answer("❌ Нет доступа", show_alert=True)
        return
    
    order_id = int(callback.data.split(":")[1])
    
    async with async_session_maker() as session:
        await update_order_status(session, callback.bot, order_id, "CONFIRMED")
    
    await callback.answer("✅ Заказ подтверждён")


@router.callback_query(F.data.startswith("order_cooking:"))
async def order_cooking(callback: CallbackQuery):
    """Set order to cooking"""
    if callback.from_user.id not in ADMIN_IDS:
        await callback.answer("❌ Нет доступа", show_alert=True)
        return
    
    order_id = int(callback.data.split(":")[1])
    
    async with async_session_maker() as session:
        await update_order_status(session, callback.bot, order_id, "COOKING")
    
    await callback.answer("🍳 Заказ готовится")


@router.callback_query(F.data.startswith("order_cancel:"))
async def order_cancel(callback: CallbackQuery):
    """Cancel order"""
    if callback.from_user.id not in ADMIN_IDS:
        await callback.answer("❌ Нет доступа", show_alert=True)
        return
    
    order_id = int(callback.data.split(":")[1])
    
    async with async_session_maker() as session:
        await update_order_status(session, callback.bot, order_id, "CANCELED")
    
    await callback.answer("❌ Заказ отменён")


@router.callback_query(F.data.startswith("order_assign_courier:"))
async def order_assign_courier(callback: CallbackQuery):
    """Show courier selection"""
    if callback.from_user.id not in ADMIN_IDS:
        await callback.answer("❌ Нет доступа", show_alert=True)
        return
    
    order_id = int(callback.data.split(":")[1])
    
    async with async_session_maker() as session:
        result = await session.execute(select(Courier).where(Courier.is_active == True))
        couriers = result.scalars().all()
        
        if not couriers:
            await callback.answer("❌ Нет активных курьеров", show_alert=True)
            return
        
        keyboard = get_courier_select_keyboard(order_id, couriers)
        await callback.message.edit_text(
            f"🚴 Выберите курьера для заказа №{order_id}",
            reply_markup=keyboard
        )
    
    await callback.answer()


@router.callback_query(F.data.startswith("assign_courier:"))
async def assign_courier_to_order(callback: CallbackQuery):
    """Assign courier to order"""
    if callback.from_user.id not in ADMIN_IDS:
        await callback.answer("❌ Нет доступа", show_alert=True)
        return
    
    _, order_id, courier_id = callback.data.split(":")
    order_id = int(order_id)
    courier_id = int(courier_id)
    
    async with async_session_maker() as session:
        await update_order_status(session, callback.bot, order_id, "COURIER_ASSIGNED", courier_id)
        
        # Send order to courier
        order = await session.get(Order, order_id)
        courier = await session.get(Courier, courier_id)
        user = await session.get(User, order.user_id)
        
        items_result = await session.execute(select(OrderItem).where(OrderItem.order_id == order_id))
        items = items_result.scalars().all()
        
        await send_order_to_courier(callback.bot, order, courier, user, items)
    
    await callback.answer(f"✅ Курьер {courier.name} назначен")
    await callback.message.edit_text("✅ Курьер назначен", reply_markup=get_admin_main_keyboard())


@router.callback_query(F.data == "admin_couriers")
async def admin_couriers(callback: CallbackQuery):
    """Courier management"""
    if callback.from_user.id not in ADMIN_IDS:
        await callback.answer("❌ Нет доступа", show_alert=True)
        return
    
    async with async_session_maker() as session:
        result = await session.execute(select(Courier))
        couriers = result.scalars().all()
        
        text = "🚴 <b>Курьеры:</b>\n\n"
        
        if couriers:
            for c in couriers:
                status = "✅" if c.is_active else "❌"
                text += f"{status} {c.name} (ID: {c.chat_id})\n"
        else:
            text += "Нет курьеров\n"
        
        text += "\nИспользуйте команды:\n/add_courier - добавить курьера"
    
    await callback.message.edit_text(text, reply_markup=get_admin_main_keyboard(), parse_mode="HTML")
    await callback.answer()


@router.message(Command("add_courier"))
async def cmd_add_courier(message: Message, state: FSMContext):
    """Start adding courier"""
    if message.from_user.id not in ADMIN_IDS:
        await message.answer("❌ Нет доступа")
        return
    
    await message.answer("Введите Telegram ID курьера:")
    await state.set_state(AdminStates.add_courier_chat_id)


@router.message(AdminStates.add_courier_chat_id)
async def add_courier_chat_id(message: Message, state: FSMContext):
    """Get courier chat ID"""
    try:
        chat_id = int(message.text)
        await state.update_data(chat_id=chat_id)
        await message.answer("Введите имя курьера:")
        await state.set_state(AdminStates.add_courier_name)
    except:
        await message.answer("❌ Неверный формат. Введите числовой ID:")


@router.message(AdminStates.add_courier_name)
async def add_courier_name(message: Message, state: FSMContext):
    """Save courier"""
    data = await state.get_data()
    
    async with async_session_maker() as session:
        courier = Courier(
            chat_id=data["chat_id"],
            name=message.text,
            is_active=True
        )
        session.add(courier)
        await session.commit()
    
    await message.answer(f"✅ Курьер {message.text} добавлен")
    await state.clear()


# ==================== COURIER HANDLERS ====================
@router.callback_query(F.data.startswith("courier_accept:"))
async def courier_accept_order(callback: CallbackQuery):
    """Courier accepts order"""
    order_id = int(callback.data.split(":")[1])
    
    async with async_session_maker() as session:
        # Verify courier
        result = await session.execute(select(Courier).where(Courier.chat_id == callback.from_user.id))
        courier = result.scalar_one_or_none()
        
        if not courier:
            await callback.answer("❌ Вы не зарегистрированы как курьер", show_alert=True)
            return
        
        await update_order_status(session, callback.bot, order_id, "OUT_FOR_DELIVERY")
    
    await callback.answer("✅ Заказ принят")
    await callback.message.edit_reply_markup(reply_markup=InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📦 Етказилди", callback_data=f"courier_delivered:{order_id}")]
    ]))


@router.callback_query(F.data.startswith("courier_delivered:"))
async def courier_delivered_order(callback: CallbackQuery):
    """Courier marks order as delivered"""
    order_id = int(callback.data.split(":")[1])
    
    async with async_session_maker() as session:
        # Verify courier
        result = await session.execute(select(Courier).where(Courier.chat_id == callback.from_user.id))
        courier = result.scalar_one_or_none()
        
        if not courier:
            await callback.answer("❌ Вы не зарегистрированы как курьер", show_alert=True)
            return
        
        await update_order_status(session, callback.bot, order_id, "DELIVERED")
    
    await callback.answer("✅ Заказ доставлен")
    await callback.message.edit_reply_markup(reply_markup=None)


# ==================== MAIN ====================
async def main():
    """Main bot function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize database
    await init_db()
    
    # Initialize bot and dispatcher
    bot = Bot(token=BOT_TOKEN)
    
    redis = Redis.from_url(REDIS_URL)
    storage = RedisStorage(redis)
    dp = Dispatcher(storage=storage)
    
    dp.include_router(router)
    
    # Start polling
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
