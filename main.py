import asyncio
import json
import hmac
import hashlib
import logging
import os
import re
import secrets
import string
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple

from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    Message,
    CallbackQuery,
    ReplyKeyboardMarkup,
    KeyboardButton,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    WebAppInfo,
)
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.redis import RedisStorage
from aiogram.exceptions import TelegramBadRequest

from sqlalchemy import (
    String, Integer, BigInteger, Boolean, DateTime, ForeignKey, Text,
    Numeric, select, func, update, UniqueConstraint
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

# =========================================================
# CONFIG
# =========================================================

@dataclass(frozen=True)
class Config:
    BOT_TOKEN: str
    ADMIN_IDS: List[int]
    DB_URL: str
    REDIS_URL: str
    SHOP_CHANNEL_ID: Optional[int]
    COURIER_CHANNEL_ID: Optional[int]
    WEBAPP_URL: str
    API_PUBLIC_BASE: str  # e.g. https://uzbke-production.up.railway.app

def parse_admin_ids(raw: str) -> List[int]:
    raw = (raw or "").strip()
    if not raw:
        return []
    parts = re.split(r"[,\s]+", raw)
    out = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(int(p))
    return out

def load_config() -> Config:
    bot_token = os.getenv("BOT_TOKEN", "").strip()
    if not bot_token:
        raise RuntimeError("BOT_TOKEN is required")

    admin_ids = parse_admin_ids(os.getenv("ADMIN_IDS", "").strip())
    if not admin_ids:
        raise RuntimeError("ADMIN_IDS is required (comma/space separated)")

    db_url = os.getenv("DB_URL", "").strip()
    if not db_url:
        raise RuntimeError("DB_URL is required")

    redis_url = os.getenv("REDIS_URL", "").strip()
    if not redis_url:
        raise RuntimeError("REDIS_URL is required")

    shop_channel_id = os.getenv("SHOP_CHANNEL_ID", "").strip()
    courier_channel_id = os.getenv("COURIER_CHANNEL_ID", "").strip()

    webapp_url = os.getenv("WEBAPP_URL", "").strip()
    if not webapp_url:
        raise RuntimeError("WEBAPP_URL is required")

    api_public_base = os.getenv("API_PUBLIC_BASE", "").strip()
    if not api_public_base:
        # fallback: if you host same process, can be blank, but WebApp needs it
        api_public_base = ""

    return Config(
        BOT_TOKEN=bot_token,
        ADMIN_IDS=admin_ids,
        DB_URL=db_url,
        REDIS_URL=redis_url,
        SHOP_CHANNEL_ID=int(shop_channel_id) if shop_channel_id else None,
        COURIER_CHANNEL_ID=int(courier_channel_id) if courier_channel_id else None,
        WEBAPP_URL=webapp_url,
        API_PUBLIC_BASE=api_public_base,
    )

CONFIG = load_config()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("fiesta")

TZ_TASHKENT = timezone(timedelta(hours=5))

# =========================================================
# DB
# =========================================================

class Base(DeclarativeBase):
    pass

class OrderStatus(str, Enum):
    NEW = "NEW"
    CONFIRMED = "CONFIRMED"
    COOKING = "COOKING"
    COURIER_ASSIGNED = "COURIER_ASSIGNED"
    OUT_FOR_DELIVERY = "OUT_FOR_DELIVERY"
    DELIVERED = "DELIVERED"
    CANCELED = "CANCELED"

STATUS_LABELS = {
    OrderStatus.NEW: "Принят",
    OrderStatus.CONFIRMED: "Подтвержден",
    OrderStatus.COOKING: "Готовится",
    OrderStatus.COURIER_ASSIGNED: "Курьер назначен",
    OrderStatus.OUT_FOR_DELIVERY: "Передан курьеру",
    OrderStatus.DELIVERED: "Доставлен",
    OrderStatus.CANCELED: "Отменен",
}

class Setting(Base):
    __tablename__ = "settings"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(String(255), nullable=False)

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tg_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    username: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    full_name: Mapped[str] = mapped_column(String(128))
    joined_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    ref_by_user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    referral_promo_issued: Mapped[bool] = mapped_column(Boolean, default=False)

class Category(Base):
    __tablename__ = "categories"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

class Food(Base):
    __tablename__ = "foods"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    category_id: Mapped[int] = mapped_column(Integer, ForeignKey("categories.id"))
    name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str] = mapped_column(String(255), default="")
    price: Mapped[int] = mapped_column(Integer)  # sum
    rating: Mapped[int] = mapped_column(Integer, default=5)
    is_new: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    image_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class Courier(Base):
    __tablename__ = "couriers"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(64))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class Promo(Base):
    __tablename__ = "promos"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    discount_percent: Mapped[int] = mapped_column(Integer)  # 1..90
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    usage_limit: Mapped[int] = mapped_column(Integer, default=0)  # 0 unlimited
    used_count: Mapped[int] = mapped_column(Integer, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

class Order(Base):
    __tablename__ = "orders"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_number: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"))
    customer_name: Mapped[str] = mapped_column(String(128))
    phone: Mapped[str] = mapped_column(String(32))
    comment: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    total: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), default=OrderStatus.NEW.value)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    delivered_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    location_lat: Mapped[float] = mapped_column(Numeric(10, 6))
    location_lng: Mapped[float] = mapped_column(Numeric(10, 6))
    courier_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("couriers.id"), nullable=True)

    admin_channel_message_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    courier_message_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

class OrderItem(Base):
    __tablename__ = "order_items"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[int] = mapped_column(Integer, ForeignKey("orders.id"))
    food_id: Mapped[int] = mapped_column(Integer, ForeignKey("foods.id"))
    name_snapshot: Mapped[str] = mapped_column(String(128))
    price_snapshot: Mapped[int] = mapped_column(Integer)
    qty: Mapped[int] = mapped_column(Integer)
    line_total: Mapped[int] = mapped_column(Integer)

    __table_args__ = (UniqueConstraint("order_id", "food_id", name="uq_order_food"),)

engine = create_async_engine(CONFIG.DB_URL, echo=False, pool_pre_ping=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

async def db_init():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with SessionLocal() as s:
        await ensure_seed_data(s)

async def ensure_seed_data(s: AsyncSession):
    # Settings: allow runtime override
    await upsert_setting(s, "SHOP_CHANNEL_ID", str(CONFIG.SHOP_CHANNEL_ID or ""))
    await upsert_setting(s, "COURIER_CHANNEL_ID", str(CONFIG.COURIER_CHANNEL_ID or ""))

    # Seed categories + 3 foods each if empty
    count = (await s.execute(select(func.count(Category.id)))).scalar_one()
    if count and count > 0:
        return

    cat_names = ["All", "Lavash", "Burger", "Xaggi", "Shaurma", "Hotdog", "Combo", "Sneki", "Sous", "Napitki"]
    cat_ids: Dict[str, int] = {}
    for name in cat_names:
        c = Category(name=name, is_active=True)
        s.add(c)
        await s.flush()
        cat_ids[name] = c.id

    demo_foods = {
        "Lavash": [
            ("Lavash Classic", "Классический лаваш", 45000, 5, True),
            ("Lavash Cheese", "С сыром", 52000, 5, False),
            ("Lavash Spicy", "Острый", 49000, 4, False),
        ],
        "Burger": [
            ("Burger Beef", "Говядина", 55000, 5, True),
            ("Burger Chicken", "Курица", 48000, 4, False),
            ("Burger Double", "Двойной", 69000, 5, False),
        ],
        "Xaggi": [
            ("Xaggi Original", "Фирменный", 60000, 5, True),
            ("Xaggi Mini", "Мини", 42000, 4, False),
            ("Xaggi XL", "Большой", 78000, 5, False),
        ],
        "Shaurma": [
            ("Shaurma Classic", "Классика", 45000, 4, True),
            ("Shaurma Cheese", "С сыром", 52000, 5, False),
            ("Shaurma Spicy", "Острая", 49000, 4, False),
        ],
        "Hotdog": [
            ("Hotdog Classic", "Классический", 30000, 4, True),
            ("Hotdog Cheese", "С сыром", 35000, 4, False),
            ("Hotdog Double", "Двойной", 42000, 5, False),
        ],
        "Combo": [
            ("Combo #1", "Бургер + фри + кола", 85000, 5, True),
            ("Combo #2", "Лаваш + фри + чай", 78000, 4, False),
            ("Combo #3", "Шаурма + напиток", 69000, 4, False),
        ],
        "Sneki": [
            ("Fries", "Картофель фри", 25000, 4, True),
            ("Nuggets", "Наггетсы", 32000, 4, False),
            ("Onion Rings", "Луковые кольца", 28000, 4, False),
        ],
        "Sous": [
            ("Ketchup", "Кетчуп", 5000, 4, True),
            ("Cheese Sauce", "Сырный соус", 7000, 5, False),
            ("Garlic Sauce", "Чесночный", 7000, 4, False),
        ],
        "Napitki": [
            ("Cola 0.5", "0.5л", 12000, 4, True),
            ("Fanta 0.5", "0.5л", 12000, 4, False),
            ("Tea", "Черный чай", 8000, 4, False),
        ],
    }

    for cat, items in demo_foods.items():
        for (name, desc, price, rating, is_new) in items:
            s.add(Food(
                category_id=cat_ids[cat],
                name=name,
                description=desc,
                price=price,
                rating=rating,
                is_new=is_new,
                is_active=True,
                image_url=None,
            ))

    await s.commit()

async def upsert_setting(s: AsyncSession, key: str, value: str):
    existing = await s.get(Setting, key)
    if existing:
        existing.value = value
    else:
        s.add(Setting(key=key, value=value))

async def get_setting_int(s: AsyncSession, key: str) -> Optional[int]:
    v = await s.get(Setting, key)
    if not v or not v.value.strip():
        return None
    try:
        return int(v.value.strip())
    except ValueError:
        return None

# =========================================================
# TELEGRAM initData VERIFY (FastAPI)
# =========================================================

def verify_telegram_init_data(init_data: str, bot_token: str) -> Dict[str, str]:
    """
    Telegram WebApp initData verification:
    https://core.telegram.org/bots/webapps#validating-data-received-via-the-mini-app
    Returns parsed key-value dict if ok, else raises.
    """
    if not init_data:
        raise ValueError("Empty initData")

    parsed = dict([kv.split("=", 1) for kv in init_data.split("&") if "=" in kv])
    if "hash" not in parsed:
        raise ValueError("No hash in initData")

    received_hash = parsed.pop("hash")

    data_check_string = "\n".join([f"{k}={parsed[k]}" for k in sorted(parsed.keys())])

    secret_key = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    calculated_hash = hmac.new(secret_key, data_check_string.encode(), hashlib.sha256).hexdigest()

    if not hmac.compare_digest(calculated_hash, received_hash):
        raise ValueError("Invalid initData hash")

    return parsed

def parse_user_from_init_data(init_data: str) -> Dict[str, Any]:
    parsed = dict([kv.split("=", 1) for kv in init_data.split("&") if "=" in kv])
    user_json = parsed.get("user")
    if not user_json:
        raise ValueError("No user in initData")
    # Telegram encodes JSON with url-encoding in initData
    # FastAPI doesn't decode it for us here; Request will pass raw header. We do percent-decoding via replace trick:
    from urllib.parse import unquote
    return json.loads(unquote(user_json))

# =========================================================
# UTIL
# =========================================================

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def fmt_dt(dt: datetime) -> str:
    # Show in Tashkent time, seconds
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    local = dt.astimezone(TZ_TASHKENT)
    return local.strftime("%Y-%m-%d %H:%M:%S")

def gen_order_number() -> str:
    # short unique
    return datetime.now().strftime("%y%m%d") + "-" + "".join(secrets.choice(string.digits) for _ in range(6))

def money(n: int) -> str:
    return f"{n:,}".replace(",", " ")

def maps_link(lat: float, lng: float) -> str:
    return f"https://maps.google.com/?q={lat},{lng}"

def is_admin(user_id: int) -> bool:
    return user_id in set(CONFIG.ADMIN_IDS)

# =========================================================
# SERVICES
# =========================================================

async def get_or_create_user(s: AsyncSession, tg_id: int, username: Optional[str], full_name: str) -> User:
    q = await s.execute(select(User).where(User.tg_id == tg_id))
    u = q.scalar_one_or_none()
    if u:
        # update basics
        u.username = username
        u.full_name = full_name
        return u
    u = User(tg_id=tg_id, username=username, full_name=full_name, joined_at=now_utc())
    s.add(u)
    await s.flush()
    return u

async def apply_referral_once(s: AsyncSession, user: User, ref_id: Optional[int]):
    if not ref_id:
        return
    if user.ref_by_user_id:
        return
    if ref_id == user.id:
        return
    ref_user = await s.get(User, ref_id)
    if not ref_user:
        return
    user.ref_by_user_id = ref_id

async def referral_stats(s: AsyncSession, user_id: int) -> Tuple[int, int, int]:
    # ref_count: users where ref_by_user_id=user_id
    ref_count = (await s.execute(select(func.count(User.id)).where(User.ref_by_user_id == user_id))).scalar_one()
    orders_count = (await s.execute(select(func.count(Order.id)).where(Order.user_id.in_(
        select(User.id).where(User.ref_by_user_id == user_id)
    )))).scalar_one()
    delivered = (await s.execute(select(func.count(Order.id)).where(
        Order.user_id.in_(select(User.id).where(User.ref_by_user_id == user_id)),
        Order.status.in_([OrderStatus.DELIVERED.value])
    ))).scalar_one()
    return int(ref_count or 0), int(orders_count or 0), int(delivered or 0)

async def maybe_issue_referral_promo(s: AsyncSession, user: User) -> Optional[Promo]:
    ref_count, _, _ = await referral_stats(s, user.id)
    if ref_count >= 3 and not user.referral_promo_issued:
        code = "FIESTA15-" + "".join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(6))
        promo = Promo(
            code=code,
            discount_percent=15,
            expires_at=now_utc() + timedelta(days=30),
            usage_limit=1,
            used_count=0,
            is_active=True,
        )
        s.add(promo)
        user.referral_promo_issued = True
        await s.flush()
        return promo
    return None

async def list_last_orders(s: AsyncSession, user_id: int, limit: int = 10) -> List[Order]:
    q = await s.execute(
        select(Order).where(Order.user_id == user_id).order_by(Order.created_at.desc()).limit(limit)
    )
    return list(q.scalars().all())

async def list_order_items(s: AsyncSession, order_id: int) -> List[OrderItem]:
    q = await s.execute(select(OrderItem).where(OrderItem.order_id == order_id))
    return list(q.scalars().all())

async def validate_promo(s: AsyncSession, code: str) -> Optional[Promo]:
    code = (code or "").strip().upper()
    if not code:
        return None
    q = await s.execute(select(Promo).where(Promo.code == code, Promo.is_active == True))
    promo = q.scalar_one_or_none()
    if not promo:
        return None
    if promo.expires_at and promo.expires_at < now_utc():
        return None
    if promo.usage_limit and promo.used_count >= promo.usage_limit:
        return None
    return promo

async def promo_mark_used(s: AsyncSession, promo_id: int):
    promo = await s.get(Promo, promo_id)
    if not promo:
        return
    promo.used_count += 1

async def create_order_from_webapp(
    s: AsyncSession,
    user: User,
    payload: Dict[str, Any],
) -> Tuple[Order, List[OrderItem], Optional[Promo]]:
    if payload.get("type") != "order_create":
        raise ValueError("Unsupported type")

    items = payload.get("items") or []
    if not isinstance(items, list) or not items:
        raise ValueError("Empty items")

    total = int(payload.get("total") or 0)
    if total < 50000:
        raise ValueError("MIN_TOTAL_50000")

    customer_name = (payload.get("customer_name") or user.full_name or "").strip()[:128]
    phone = (payload.get("phone") or "").strip()[:32]
    comment = (payload.get("comment") or "").strip()[:255] or None

    loc = payload.get("location") or {}
    lat = loc.get("lat")
    lng = loc.get("lng")
    if lat is None or lng is None:
        raise ValueError("NO_LOCATION")

    promo_code = (payload.get("promo_code") or "").strip().upper()
    promo = None
    if promo_code:
        promo = await validate_promo(s, promo_code)
        if not promo:
            raise ValueError("INVALID_PROMO")

    order_number = gen_order_number()
    order = Order(
        order_number=order_number,
        user_id=user.id,
        customer_name=customer_name or user.full_name,
        phone=phone,
        comment=comment,
        total=total,
        status=OrderStatus.NEW.value,
        created_at=now_utc(),
        updated_at=now_utc(),
        delivered_at=None,
        location_lat=float(lat),
        location_lng=float(lng),
        courier_id=None,
        admin_channel_message_id=None,
        courier_message_id=None,
    )
    s.add(order)
    await s.flush()

    out_items: List[OrderItem] = []
    # trust snapshot from payload, but verify food_id exists
    for it in items:
        food_id = int(it.get("food_id"))
        qty = int(it.get("qty"))
        if qty <= 0:
            continue
        food = await s.get(Food, food_id)
        if not food or not food.is_active:
            raise ValueError(f"FOOD_NOT_FOUND:{food_id}")

        name_snap = str(it.get("name") or food.name)[:128]
        price_snap = int(it.get("price") or food.price)
        line_total = price_snap * qty
        oi = OrderItem(
            order_id=order.id,
            food_id=food_id,
            name_snapshot=name_snap,
            price_snapshot=price_snap,
            qty=qty,
            line_total=line_total
        )
        s.add(oi)
        out_items.append(oi)

    if not out_items:
        raise ValueError("Empty items after validation")

    if promo:
        # We do not change total here (client already computed). Just mark used on success.
        await promo_mark_used(s, promo.id)

    await s.commit()
    return order, out_items, promo

async def update_order_status(s: AsyncSession, order_id: int, new_status: OrderStatus, courier_id: Optional[int] = None):
    o = await s.get(Order, order_id)
    if not o:
        raise ValueError("ORDER_NOT_FOUND")
    o.status = new_status.value
    o.updated_at = now_utc()
    if courier_id is not None:
        o.courier_id = courier_id
    if new_status == OrderStatus.DELIVERED:
        o.delivered_at = now_utc()
    await s.commit()
    return o

async def active_orders(s: AsyncSession) -> List[Order]:
    active = [OrderStatus.NEW.value, OrderStatus.CONFIRMED.value, OrderStatus.COOKING.value,
              OrderStatus.COURIER_ASSIGNED.value, OrderStatus.OUT_FOR_DELIVERY.value]
    q = await s.execute(select(Order).where(Order.status.in_(active)).order_by(Order.created_at.desc()))
    return list(q.scalars().all())

async def get_bot_username(bot: Bot) -> str:
    me = await bot.get_me()
    return me.username or "YourBot"

# =========================================================
# TELEGRAM NOTIFY / MESSAGE BUILDERS
# =========================================================

def order_items_text(items: List[OrderItem]) -> str:
    lines = []
    for it in items:
        lines.append(f"• {it.name_snapshot} x{it.qty} = {money(it.line_total)}")
    return "\n".join(lines)

def admin_order_post_text(user: User, order: Order, items: List[OrderItem]) -> str:
    lat = float(order.location_lat)
    lng = float(order.location_lng)
    return (
        f"🆕 Новый заказ №{order.order_number}\n"
        f"👤 Пользователь: {user.full_name} (@{user.username or '—'})\n"
        f"📞 Телефон: {order.phone}\n"
        f"💰 Сумма: {money(order.total)} сум\n"
        f"🕒 Время: {fmt_dt(order.created_at)}\n"
        f"📍 Локация: {lat},{lng}\n"
        f"🔗 Карта: {maps_link(lat,lng)}\n"
        f"🍽️ Заказ:\n{order_items_text(items)}\n\n"
        f"📦 Статус: {STATUS_LABELS[OrderStatus(order.status)]}"
    )

def user_order_accept_text(order: Order) -> str:
    return (
        f"Ваш заказ принят ✅\n"
        f"🆔 Заказ №{order.order_number}\n"
        f"💰 Сумма: {money(order.total)} сум\n"
        f"📦 Статус: {STATUS_LABELS[OrderStatus(order.status)]}"
    )

def courier_order_text(order: Order, items: List[OrderItem]) -> str:
    lat = float(order.location_lat)
    lng = float(order.location_lng)
    return (
        f"🚴 Новый заказ №{order.order_number}\n"
        f"👤 Клиент: {order.customer_name}\n"
        f"📞 Телефон: {order.phone}\n"
        f"💰 Сумма: {money(order.total)} сум\n"
        f"📍 Локация: {maps_link(lat,lng)}\n"
        f"🍽️ Список:\n{order_items_text(items)}"
    )

def admin_inline_kb_for_order(order_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="✅ Подтвержден", callback_data=f"adm_status:{order_id}:CONFIRMED"),
            InlineKeyboardButton(text="🍳 Готовится", callback_data=f"adm_status:{order_id}:COOKING"),
        ],
        [
            InlineKeyboardButton(text="🚴 Курьер", callback_data=f"adm_courier_pick:{order_id}"),
            InlineKeyboardButton(text="❌ Отменен", callback_data=f"adm_status:{order_id}:CANCELED"),
        ],
    ])

def courier_kb(order_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="✅ Qabul qildim", callback_data=f"courier_accept:{order_id}"),
            InlineKeyboardButton(text="📦 Yetkazildi", callback_data=f"courier_delivered:{order_id}"),
        ]
    ])

# =========================================================
# KEYBOARDS (CLIENT)
# =========================================================

def client_main_kb(webapp_url: str) -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🛍 Заказать", web_app=WebAppInfo(url=webapp_url))],
            [KeyboardButton(text="📦 Мои заказы")],
            [KeyboardButton(text="ℹ️ Информация о нас")],
            [KeyboardButton(text="👥 Пригласить друга")],
        ],
        resize_keyboard=True
    )

def shop_inline_kb(webapp_url: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🛍 Заказать", web_app=WebAppInfo(url=webapp_url))]
    ])

# =========================================================
# FSM (ADMIN)
# =========================================================

class AdminStates(StatesGroup):
    set_shop_channel = State()
    set_courier_channel = State()
    add_courier_chat_id = State()
    add_courier_name = State()
    create_promo_code = State()
    create_promo_discount = State()
    create_promo_expires_days = State()
    create_promo_usage_limit = State()

# =========================================================
# BOT ROUTERS
# =========================================================

router_client = Router()
router_admin = Router()
router_courier = Router()

# ---------------- CLIENT: /start ----------------

@router_client.message(CommandStart())
async def cmd_start(message: Message):
    ref_arg = None
    if message.text:
        parts = message.text.strip().split(maxsplit=1)
        if len(parts) == 2:
            ref_arg = parts[1].strip()
    ref_id = None
    if ref_arg and ref_arg.isdigit():
        ref_id = int(ref_arg)

    async with SessionLocal() as s:
        u = await get_or_create_user(
            s,
            tg_id=message.from_user.id,
            username=message.from_user.username,
            full_name=message.from_user.full_name or "User"
        )
        await apply_referral_once(s, u, ref_id)
        await s.commit()

    await message.answer(
        f"Добро пожаловать в FIESTA! {message.from_user.full_name}\n"
        f"Для заказа перейдите по кнопке ➡️\n"
        f"🛍 Заказать",
        reply_markup=client_main_kb(CONFIG.WEBAPP_URL)
    )

# ---------------- CLIENT: menu buttons ----------------

@router_client.message(F.text == "ℹ️ Информация о нас")
async def info_about(message: Message):
    await message.answer(
        "🌟 Добро Пожаловать в FIESTA !\n"
        "📍 Наш адрес:Хорезмская область, г.Хива, махаллинский сход граждан Гиламчи\n"
        "🏢﻿ Ориентир: Школа №12 Оруджева\n"
        "📞 Контактный номер: +998 91 420 15 15\n"
        "🕙﻿ Рабочие часы: 24/7\n"
        "📷 Мы в Instagram: fiesta.khiva (https://www.instagram.com/fiesta.khiva?igsh=Z3VoMzE0eGx0ZTVo)\n"
        "🔗 Найти нас на карте: Место расположение (https://maps.app.goo.gl/dpBVHBWX1K7NTYVR7)"
    )

@router_client.message(F.text == "📦 Мои заказы")
async def my_orders(message: Message):
    async with SessionLocal() as s:
        u_q = await s.execute(select(User).where(User.tg_id == message.from_user.id))
        u = u_q.scalar_one_or_none()
        if not u:
            await message.answer("Сначала нажмите /start")
            return

        orders = await list_last_orders(s, u.id, limit=10)
        if not orders:
            await message.answer(
                "В данный момент у вас нет активных заказов в нашем магазине.\n"
                "Чтобы открыть магазин, введите команду — /shop"
            )
            return

        texts = []
        for o in orders:
            items = await list_order_items(s, o.id)
            head = f"🆔 Заказ №{o.order_number} | {fmt_dt(o.created_at)} | 💰 {money(o.total)} | 📦 {STATUS_LABELS[OrderStatus(o.status)]}"
            body = "\n".join([f"  - {it.name_snapshot} x{it.qty} = {money(it.line_total)}" for it in items])
            texts.append(head + "\n" + body)
        await message.answer("\n\n".join(texts))

@router_client.message(Command("shop"))
async def shop_cmd(message: Message):
    await message.answer(
        "Чтобы открыть наш магазин, нажмите кнопку ниже",
        reply_markup=shop_inline_kb(CONFIG.WEBAPP_URL)
    )

@router_client.message(F.text == "👥 Пригласить друга")
async def invite_friend(message: Message, bot: Bot):
    async with SessionLocal() as s:
        u_q = await s.execute(select(User).where(User.tg_id == message.from_user.id))
        u = u_q.scalar_one_or_none()
        if not u:
            await message.answer("Сначала нажмите /start")
            return
        bot_username = await get_bot_username(bot)
        ref_count, orders_count, delivered_count = await referral_stats(s, u.id)
        promo = await maybe_issue_referral_promo(s, u)
        await s.commit()

    await message.answer(
        "За приглашение друга, вы можете получить промо-код от нас\n"
        f"👥 Вы пригласили {ref_count} человек\n"
        f"🛒 Оформили заказов: {orders_count}\n"
        f"💰 Оплатили заказов: {delivered_count}\n"
        f"👤 Ваша реферальная ссылка: https://t.me/{bot_username}?start={u.id}\n"
        "Пригласите трех человек и вы получите от нас промо-код со скидкой 15%"
    )
    if promo:
        await message.answer(
            f"🎁 Ваш промокод на 15% готов!\n"
            f"Код: {promo.code}\n"
            f"Действует до: {fmt_dt(promo.expires_at)}\n"
            f"Использований: 1"
        )

# ---------------- WEBAPP -> BOT ----------------

@router_client.message(F.web_app_data)
async def webapp_data(message: Message, bot: Bot):
    try:
        payload = json.loads(message.web_app_data.data)
    except Exception:
        await message.answer("Ошибка данных WebApp. Попробуйте ещё раз.")
        return

    async with SessionLocal() as s:
        u_q = await s.execute(select(User).where(User.tg_id == message.from_user.id))
        u = u_q.scalar_one_or_none()
        if not u:
            u = await get_or_create_user(s, message.from_user.id, message.from_user.username, message.from_user.full_name or "User")
            await s.commit()

        try:
            order, items, promo = await create_order_from_webapp(s, u, payload)
        except ValueError as e:
            code = str(e)
            if code == "MIN_TOTAL_50000":
                await message.answer("Минимальная сумма заказа: 50 000 сум.")
            elif code == "NO_LOCATION":
                await message.answer("Нужно отправить локацию (lat/lng).")
            elif code == "INVALID_PROMO":
                await message.answer("Промокод недействителен.")
            else:
                await message.answer(f"Ошибка: {code}")
            return
        except Exception as e:
            logger.exception("create_order error")
            await message.answer("Ошибка создания заказа. Попробуйте позже.")
            return

        await message.answer(user_order_accept_text(order))

        # Notify admin channel
        shop_channel_id = await get_setting_int(s, "SHOP_CHANNEL_ID")
        if shop_channel_id:
            txt = admin_order_post_text(u, order, items)
            msg = await bot.send_message(
                chat_id=shop_channel_id,
                text=txt,
                reply_markup=admin_inline_kb_for_order(order.id)
            )
            order.admin_channel_message_id = msg.message_id
            order.updated_at = now_utc()
            await s.commit()

# =========================================================
# ADMIN PANEL
# =========================================================

def admin_menu_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📦 Актив buyurtmalar", callback_data="adm_active_orders")],
        [InlineKeyboardButton(text="🚴 Kuryerlar", callback_data="adm_couriers")],
        [InlineKeyboardButton(text="🎁 Promokodlar", callback_data="adm_promos")],
        [InlineKeyboardButton(text="⚙️ Sozlamalar", callback_data="adm_settings")],
    ])

@router_admin.message(Command("admin"))
async def admin_cmd(message: Message):
    if not is_admin(message.from_user.id):
        return
    await message.answer("🔧 Admin panel", reply_markup=admin_menu_kb())

@router_admin.callback_query(F.data == "adm_settings")
async def adm_settings(cb: CallbackQuery):
    if not is_admin(cb.from_user.id):
        return
    await cb.message.edit_text(
        "⚙️ Sozlamalar:\n"
        "SHOP_CHANNEL_ID va COURIER_CHANNEL_ID ni bot ichida o‘rnating.\n\n"
        "Tanlang:",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🏪 SHOP_CHANNEL_ID set", callback_data="adm_set_shop")],
            [InlineKeyboardButton(text="🚴 COURIER_CHANNEL_ID set", callback_data="adm_set_courier_channel")],
            [InlineKeyboardButton(text="⬅️ Back", callback_data="adm_back")],
        ])
    )
    await cb.answer()

@router_admin.callback_query(F.data == "adm_back")
async def adm_back(cb: CallbackQuery):
    if not is_admin(cb.from_user.id):
        return
    await cb.message.edit_text("🔧 Admin panel", reply_markup=admin_menu_kb())
    await cb.answer()

@router_admin.callback_query(F.data == "adm_set_shop")
async def adm_set_shop(cb: CallbackQuery, state: FSMContext):
    if not is_admin(cb.from_user.id):
        return
    await state.set_state(AdminStates.set_shop_channel)
    await cb.message.answer("SHOP_CHANNEL_ID yuboring (masalan: -1001234567890) yoki 0 (o‘chirish).")
    await cb.answer()

@router_admin.message(AdminStates.set_shop_channel)
async def adm_set_shop_value(message: Message, state: FSMContext):
    if not is_admin(message.from_user.id):
        return
    raw = (message.text or "").strip()
    if not re.fullmatch(r"-?\d+", raw):
        await message.answer("Faqat raqam. Masalan: -1001234567890 yoki 0")
        return
    val = "" if raw == "0" else raw
    async with SessionLocal() as s:
        await upsert_setting(s, "SHOP_CHANNEL_ID", val)
        await s.commit()
    await state.clear()
    await message.answer(f"✅ SHOP_CHANNEL_ID set: {val or '(disabled)'}")

@router_admin.callback_query(F.data == "adm_set_courier_channel")
async def adm_set_courier_channel(cb: CallbackQuery, state: FSMContext):
    if not is_admin(cb.from_user.id):
        return
    await state.set_state(AdminStates.set_courier_channel)
    await cb.message.answer("COURIER_CHANNEL_ID yuboring (masalan: -1001234567890) yoki 0 (o‘chirish).")
    await cb.answer()

@router_admin.message(AdminStates.set_courier_channel)
async def adm_set_courier_channel_value(message: Message, state: FSMContext):
    if not is_admin(message.from_user.id):
        return
    raw = (message.text or "").strip()
    if not re.fullmatch(r"-?\d+", raw):
        await message.answer("Faqat raqam. Masalan: -1001234567890 yoki 0")
        return
    val = "" if raw == "0" else raw
    async with SessionLocal() as s:
        await upsert_setting(s, "COURIER_CHANNEL_ID", val)
        await s.commit()
    await state.clear()
    await message.answer(f"✅ COURIER_CHANNEL_ID set: {val or '(disabled)'}")

# -------- Admin: active orders list --------

@router_admin.callback_query(F.data == "adm_active_orders")
async def adm_active_orders(cb: CallbackQuery):
    if not is_admin(cb.from_user.id):
        return
    async with SessionLocal() as s:
        orders = await active_orders(s)
        if not orders:
            await cb.message.edit_text("Актив buyurtmalar yo‘q.", reply_markup=admin_menu_kb())
            await cb.answer()
            return
        lines = []
        for o in orders[:25]:
            lines.append(f"🆔 {o.order_number} | {money(o.total)} | {STATUS_LABELS[OrderStatus(o.status)]}")
        await cb.message.edit_text(
            "📦 Aktiv buyurtmalar:\n" + "\n".join(lines),
            reply_markup=admin_menu_kb()
        )
    await cb.answer()

# -------- Admin: couriers manage (add/list/toggle) --------

@router_admin.callback_query(F.data == "adm_couriers")
async def adm_couriers(cb: CallbackQuery):
    if not is_admin(cb.from_user.id):
        return
    async with SessionLocal() as s:
        q = await s.execute(select(Courier).order_by(Courier.created_at.desc()))
        couriers = list(q.scalars().all())

    kb_rows = [[InlineKeyboardButton(text="➕ Add courier", callback_data="adm_add_courier")]]
    for c in couriers[:20]:
        status = "✅" if c.is_active else "⛔"
        kb_rows.append([
            InlineKeyboardButton(text=f"{status} {c.name} ({c.chat_id})", callback_data=f"adm_toggle_courier:{c.id}")
        ])
    kb_rows.append([InlineKeyboardButton(text="⬅️ Back", callback_data="adm_back")])

    await cb.message.edit_text("🚴 Kuryerlar:", reply_markup=InlineKeyboardMarkup(inline_keyboard=kb_rows))
    await cb.answer()

@router_admin.callback_query(F.data == "adm_add_courier")
async def adm_add_courier(cb: CallbackQuery, state: FSMContext):
    if not is_admin(cb.from_user.id):
        return
    await state.set_state(AdminStates.add_courier_chat_id)
    await cb.message.answer("Kuryer chat_id yuboring (masalan: 123456789).")
    await cb.answer()

@router_admin.message(AdminStates.add_courier_chat_id)
async def adm_add_courier_chat_id(message: Message, state: FSMContext):
    if not is_admin(message.from_user.id):
        return
    raw = (message.text or "").strip()
    if not re.fullmatch(r"-?\d+", raw):
        await message.answer("Faqat raqam chat_id.")
        return
    await state.update_data(chat_id=int(raw))
    await state.set_state(AdminStates.add_courier_name)
    await message.answer("Kuryer nomini yuboring (masalan: Bobur).")

@router_admin.message(AdminStates.add_courier_name)
async def adm_add_courier_name(message: Message, state: FSMContext):
    if not is_admin(message.from_user.id):
        return
    name = (message.text or "").strip()[:64]
    data = await state.get_data()
    chat_id = int(data["chat_id"])
    async with SessionLocal() as s:
        q = await s.execute(select(Courier).where(Courier.chat_id == chat_id))
        existing = q.scalar_one_or_none()
        if existing:
            existing.name = name
            existing.is_active = True
        else:
            s.add(Courier(chat_id=chat_id, name=name, is_active=True))
        await s.commit()
    await state.clear()
    await message.answer("✅ Kuryer qo‘shildi/yangilandi. /admin -> Kuryerlar")

@router_admin.callback_query(F.data.startswith("adm_toggle_courier:"))
async def adm_toggle_courier(cb: CallbackQuery):
    if not is_admin(cb.from_user.id):
        return
    cid = int(cb.data.split(":")[1])
    async with SessionLocal() as s:
        c = await s.get(Courier, cid)
        if c:
            c.is_active = not c.is_active
            await s.commit()
    await cb.answer("OK")
    # refresh
    await adm_couriers(cb)

# -------- Admin: promos create/list --------

@router_admin.callback_query(F.data == "adm_promos")
async def adm_promos(cb: CallbackQuery):
    if not is_admin(cb.from_user.id):
        return
    async with SessionLocal() as s:
        q = await s.execute(select(Promo).order_by(Promo.id.desc()).limit(20))
        promos = list(q.scalars().all())
    lines = []
    for p in promos:
        exp = fmt_dt(p.expires_at) if p.expires_at else "∞"
        lim = p.usage_limit if p.usage_limit else "∞"
        lines.append(f"• {p.code} | -{p.discount_percent}% | exp: {exp} | used: {p.used_count}/{lim}")
    await cb.message.edit_text(
        "🎁 Promokodlar:\n" + ("\n".join(lines) if lines else "Hozircha yo‘q."),
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="➕ Create promo", callback_data="adm_create_promo")],
            [InlineKeyboardButton(text="⬅️ Back", callback_data="adm_back")],
        ])
    )
    await cb.answer()

@router_admin.callback_query(F.data == "adm_create_promo")
async def adm_create_promo(cb: CallbackQuery, state: FSMContext):
    if not is_admin(cb.from_user.id):
        return
    await state.set_state(AdminStates.create_promo_code)
    await cb.message.answer("Promo code yuboring (masalan: FIESTA10).")
    await cb.answer()

@router_admin.message(AdminStates.create_promo_code)
async def adm_create_promo_code(message: Message, state: FSMContext):
    if not is_admin(message.from_user.id):
        return
    code = (message.text or "").strip().upper()
    if not re.fullmatch(r"[A-Z0-9_\-]{4,32}", code):
        await message.answer("Code formati xato. (A-Z0-9_-), 4..32")
        return
    await state.update_data(code=code)
    await state.set_state(AdminStates.create_promo_discount)
    await message.answer("Discount % yuboring (1..90).")

@router_admin.message(AdminStates.create_promo_discount)
async def adm_create_promo_discount(message: Message, state: FSMContext):
    if not is_admin(message.from_user.id):
        return
    raw = (message.text or "").strip()
    if not raw.isdigit():
        await message.answer("Faqat raqam 1..90")
        return
    disc = int(raw)
    if disc < 1 or disc > 90:
        await message.answer("1..90 oralig‘ida.")
        return
    await state.update_data(discount=disc)
    await state.set_state(AdminStates.create_promo_expires_days)
    await message.answer("Necha kunga amal qiladi? (masalan 30) yoki 0 (cheksiz).")

@router_admin.message(AdminStates.create_promo_expires_days)
async def adm_create_promo_expires(message: Message, state: FSMContext):
    if not is_admin(message.from_user.id):
        return
    raw = (message.text or "").strip()
    if not raw.isdigit():
        await message.answer("Faqat raqam.")
        return
    days = int(raw)
    await state.update_data(expires_days=days)
    await state.set_state(AdminStates.create_promo_usage_limit)
    await message.answer("Usage limit? (masalan 10) yoki 0 (cheksiz).")

@router_admin.message(AdminStates.create_promo_usage_limit)
async def adm_create_promo_limit(message: Message, state: FSMContext):
    if not is_admin(message.from_user.id):
        return
    raw = (message.text or "").strip()
    if not raw.isdigit():
        await message.answer("Faqat raqam.")
        return
    limit = int(raw)
    data = await state.get_data()
    code = data["code"]
    disc = int(data["discount"])
    days = int(data["expires_days"])

    expires_at = None if days == 0 else now_utc() + timedelta(days=days)
    async with SessionLocal() as s:
        q = await s.execute(select(Promo).where(Promo.code == code))
        if q.scalar_one_or_none():
            await message.answer("Bu code allaqachon bor.")
            return
        s.add(Promo(code=code, discount_percent=disc, expires_at=expires_at, usage_limit=limit, used_count=0, is_active=True))
        await s.commit()

    await state.clear()
    await message.answer("✅ Promo yaratildi. /admin -> Promokodlar")

# =========================================================
# ADMIN STATUS CALLBACKS + COURIER ASSIGN FLOW
# =========================================================

async def edit_admin_post(bot: Bot, shop_channel_id: int, order: Order, user: User, items: List[OrderItem]):
    if not order.admin_channel_message_id:
        return
    try:
        await bot.edit_message_text(
            chat_id=shop_channel_id,
            message_id=order.admin_channel_message_id,
            text=admin_order_post_text(user, order, items),
            reply_markup=(None if OrderStatus(order.status) in [OrderStatus.DELIVERED, OrderStatus.CANCELED] else admin_inline_kb_for_order(order.id))
        )
    except TelegramBadRequest:
        # message might be too old or not editable; ignore
        pass

@router_admin.callback_query(F.data.startswith("adm_status:"))
async def adm_set_status(cb: CallbackQuery, bot: Bot):
    if not is_admin(cb.from_user.id):
        return

    _, order_id_s, status_s = cb.data.split(":")
    order_id = int(order_id_s)
    new_status = OrderStatus(status_s)

    async with SessionLocal() as s:
        order = await update_order_status(s, order_id, new_status)
        user = await s.get(User, order.user_id)
        items = await list_order_items(s, order.id)

        shop_channel_id = await get_setting_int(s, "SHOP_CHANNEL_ID")
        if shop_channel_id:
            await edit_admin_post(bot, shop_channel_id, order, user, items)

        # user notify
        try:
            await bot.send_message(
                chat_id=user.tg_id,
                text=f"Статус заказа №{order.order_number} обновлён ✅\n📦 {STATUS_LABELS[new_status]}"
            )
        except Exception:
            pass

    await cb.answer("OK")

@router_admin.callback_query(F.data.startswith("adm_courier_pick:"))
async def adm_pick_courier(cb: CallbackQuery):
    if not is_admin(cb.from_user.id):
        return
    order_id = int(cb.data.split(":")[1])

    async with SessionLocal() as s:
        q = await s.execute(select(Courier).where(Courier.is_active == True).order_by(Courier.name.asc()))
        couriers = list(q.scalars().all())

    if not couriers:
        await cb.answer("Active kuryer yo‘q. /admin -> Kuryerlar -> Add", show_alert=True)
        return

    rows = []
    for c in couriers:
        rows.append([InlineKeyboardButton(text=f"🚴 {c.name}", callback_data=f"adm_assign_courier:{order_id}:{c.id}")])
    rows.append([InlineKeyboardButton(text="⬅️ Back", callback_data="adm_back")])

    await cb.message.answer(
        f"Выберите курьера для заказа (ID={order_id})",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=rows)
    )
    await cb.answer()

@router_admin.callback_query(F.data.startswith("adm_assign_courier:"))
async def adm_assign_courier(cb: CallbackQuery, bot: Bot):
    if not is_admin(cb.from_user.id):
        return
    _, order_id_s, courier_id_s = cb.data.split(":")
    order_id = int(order_id_s)
    courier_id = int(courier_id_s)

    async with SessionLocal() as s:
        courier = await s.get(Courier, courier_id)
        if not courier or not courier.is_active:
            await cb.answer("Courier not found", show_alert=True)
            return

        order = await update_order_status(s, order_id, OrderStatus.COURIER_ASSIGNED, courier_id=courier_id)
        user = await s.get(User, order.user_id)
        items = await list_order_items(s, order.id)

        courier_channel_id = await get_setting_int(s, "COURIER_CHANNEL_ID")
        shop_channel_id = await get_setting_int(s, "SHOP_CHANNEL_ID")

        # Send to courier (priority: personal chat_id; additionally optional courier channel)
        text = courier_order_text(order, items)
        sent_msg = await bot.send_message(chat_id=courier.chat_id, text=text, reply_markup=courier_kb(order.id))
        order.courier_message_id = sent_msg.message_id
        await s.commit()

        if courier_channel_id:
            # log copy (not required to have buttons)
            await bot.send_message(chat_id=courier_channel_id, text=text)

        # edit admin post + user notify
        if shop_channel_id:
            await edit_admin_post(bot, shop_channel_id, order, user, items)

        try:
            await bot.send_message(chat_id=user.tg_id, text=f"Ваш заказ №{order.order_number} назначен курьеру 🚴")
        except Exception:
            pass

    await cb.answer("Courier assigned ✅")

# =========================================================
# COURIER HANDLERS (only registered couriers)
# =========================================================

async def courier_allowed(s: AsyncSession, tg_id: int) -> Optional[Courier]:
    q = await s.execute(select(Courier).where(Courier.chat_id == tg_id, Courier.is_active == True))
    return q.scalar_one_or_none()

@router_courier.callback_query(F.data.startswith("courier_accept:"))
async def courier_accept(cb: CallbackQuery, bot: Bot):
    order_id = int(cb.data.split(":")[1])
    async with SessionLocal() as s:
        courier = await courier_allowed(s, cb.from_user.id)
        if not courier:
            await cb.answer("Siz kuryer emassiz.", show_alert=True)
            return

        order = await s.get(Order, order_id)
        if not order or order.courier_id != courier.id:
            await cb.answer("Order topilmadi yoki sizga tegishli emas.", show_alert=True)
            return

        order = await update_order_status(s, order_id, OrderStatus.OUT_FOR_DELIVERY)
        user = await s.get(User, order.user_id)
        items = await list_order_items(s, order.id)

        shop_channel_id = await get_setting_int(s, "SHOP_CHANNEL_ID")
        if shop_channel_id:
            await edit_admin_post(bot, shop_channel_id, order, user, items)

        try:
            await bot.send_message(chat_id=user.tg_id, text=f"Ваш заказ №{order.order_number} передан курьеру 🚴")
        except Exception:
            pass

    await cb.answer("Qabul qilindi ✅")

@router_courier.callback_query(F.data.startswith("courier_delivered:"))
async def courier_delivered(cb: CallbackQuery, bot: Bot):
    order_id = int(cb.data.split(":")[1])
    async with SessionLocal() as s:
        courier = await courier_allowed(s, cb.from_user.id)
        if not courier:
            await cb.answer("Siz kuryer emassiz.", show_alert=True)
            return

        order = await s.get(Order, order_id)
        if not order or order.courier_id != courier.id:
            await cb.answer("Order topilmadi yoki sizga tegishli emas.", show_alert=True)
            return

        order = await update_order_status(s, order_id, OrderStatus.DELIVERED)
        user = await s.get(User, order.user_id)
        items = await list_order_items(s, order.id)

        shop_channel_id = await get_setting_int(s, "SHOP_CHANNEL_ID")
        if shop_channel_id:
            await edit_admin_post(bot, shop_channel_id, order, user, items)

        # remove courier buttons
        try:
            await cb.message.edit_reply_markup(reply_markup=None)
        except Exception:
            pass

        try:
            await bot.send_message(chat_id=user.tg_id, text=f"Ваш заказ №{order.order_number} успешно доставлен 🎉 Спасибо!")
        except Exception:
            pass

    await cb.answer("Yetkazildi ✅")

# =========================================================
# FASTAPI (foods/categories/promo validate)
# =========================================================

api = FastAPI(title="FIESTA API", version="1.0.0")
app = api
async def require_tg_user(init_data: str) -> Dict[str, Any]:
    try:
        verify_telegram_init_data(init_data, CONFIG.BOT_TOKEN)
        user = parse_user_from_init_data(init_data)
        return user
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"initData invalid: {e}")

@api.get("/api/health")
async def health():
    return {"ok": True, "ts": fmt_dt(now_utc())}

@api.get("/api/categories")
async def api_categories(x_tg_init_data: str = Header(default="")):
    await require_tg_user(x_tg_init_data)
    async with SessionLocal() as s:
        q = await s.execute(select(Category).where(Category.is_active == True).order_by(Category.name.asc()))
        cats = [{"id": c.id, "name": c.name} for c in q.scalars().all()]
    return {"categories": cats}

@api.get("/api/foods")
async def api_foods(
    x_tg_init_data: str = Header(default=""),
    category_id: Optional[int] = None,
    q: Optional[str] = None,
    sort: Optional[str] = None,  # rating_desc, created_desc, price_asc, price_desc
):
    await require_tg_user(x_tg_init_data)
    async with SessionLocal() as s:
        stmt = select(Food, Category).join(Category, Food.category_id == Category.id).where(Food.is_active == True, Category.is_active == True)

        if category_id:
            stmt = stmt.where(Food.category_id == category_id)
        if q:
            like = f"%{q.strip()}%"
            stmt = stmt.where(Food.name.ilike(like))

        if sort == "rating_desc":
            stmt = stmt.order_by(Food.rating.desc(), Food.id.desc())
        elif sort == "created_desc":
            stmt = stmt.order_by(Food.created_at.desc())
        elif sort == "price_asc":
            stmt = stmt.order_by(Food.price.asc())
        elif sort == "price_desc":
            stmt = stmt.order_by(Food.price.desc())
        else:
            stmt = stmt.order_by(Food.id.desc())

        res = await s.execute(stmt.limit(300))
        foods = []
        for f, c in res.all():
            foods.append({
                "id": f.id,
                "category_id": f.category_id,
                "category_name": c.name,
                "name": f.name,
                "description": f.description,
                "price": f.price,
                "rating": f.rating,
                "is_new": f.is_new,
                "image_url": f.image_url,
                "created_at": f.created_at.isoformat(),
            })
    return {"foods": foods}

@api.get("/api/promo/validate")
async def api_promo_validate(x_tg_init_data: str = Header(default=""), code: str = ""):
    await require_tg_user(x_tg_init_data)
    async with SessionLocal() as s:
        promo = await validate_promo(s, code)
        if not promo:
            return {"ok": False}
        return {
            "ok": True,
            "code": promo.code,
            "discount_percent": promo.discount_percent,
            "expires_at": promo.expires_at.isoformat() if promo.expires_at else None,
            "usage_limit": promo.usage_limit,
            "used_count": promo.used_count,
        }

# =========================================================
# BOOTSTRAP: run bot + api in one process
# =========================================================

async def start_bot():
    bot = Bot(CONFIG.BOT_TOKEN)
    storage = RedisStorage.from_url(CONFIG.REDIS_URL)
    dp = Dispatcher(storage=storage)
    dp.include_router(router_client)
    dp.include_router(router_admin)
    dp.include_router(router_courier)

    await bot.delete_webhook(drop_pending_updates=True)
    logger.info("Bot polling started")
    await dp.start_polling(bot)

async def start_api():
    # If you run on Railway, they usually set PORT
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    config = uvicorn.Config(api, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    logger.info("API server started")
    await server.serve()

async def main():
    await db_init()
    # Run both concurrently:
    await asyncio.gather(
        start_api(),
        start_bot(),
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
