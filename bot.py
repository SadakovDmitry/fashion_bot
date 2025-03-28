from aiogram.types import BotCommand
import random
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.utils import executor
from PIL import Image
import io
import os
import glob

# Токен бота
TOKEN = "7803258791:AAE1sFkqfQyQjeea-E1TImzCI4z6d9B5xuk"

# Создаём бота и диспетчер
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

async def set_commands():
    commands = [
        BotCommand(command="/start", description="Запустить бота"),
    ]
    await bot.set_my_commands(commands)

# Клавиатура с кнопками
keyboard = ReplyKeyboardMarkup(resize_keyboard=True)
keyboard.add(KeyboardButton("/start"))
keyboard.add(KeyboardButton("Собрать образ"))
keyboard.add(KeyboardButton("Посмотреть гардероб"))
keyboard.add(KeyboardButton("Очистить гардероб"))

# Загрузка предобученной модели MobileNetV3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
model.classifier[3] = torch.nn.Linear(in_features=1024, out_features=2)  # Верх (0) / Низ (1)
model.load_state_dict(torch.load("clothing_model.pth", map_location=device))  # Загрузка обученных весов
model.eval().to(device)

# Трансформации для изображения
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Папки для хранения изображений
os.makedirs("images/top", exist_ok=True)
os.makedirs("images/bottom", exist_ok=True)

def classify_image(image: Image.Image) -> str:
    """Классифицирует изображение как верх (1) или низ (0)."""
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()
    return "верх" if prediction == 1 else "низ"

@dp.message_handler(commands=['start'])
async def start(message: types.Message):
    await message.answer(
        "Привет! Загрузи фото одежды, и я разберу, что это.\n\n"
        "📌 Используй кнопки:\n"
        "✅ /start – Перезапустить бота\n"
        "✅ Собрать образ – Случайный комплект\n"
        "✅ Посмотреть гардероб – Все загруженные вещи\n"
        "✅ Очистить гардероб – Удалить все вещи",
        reply_markup=keyboard
    )

@dp.message_handler(content_types=['photo'])
async def handle_photo(message: types.Message):
    """Обрабатываем загруженные фото, классифицируем и сохраняем."""
    os.makedirs("images/temp", exist_ok=True)  # Убеждаемся, что папка существует

    # Получаем путь для сохранения фото
    filename = f"{message.photo[-1].file_id}.jpg"
    image_path = f"images/temp/{filename}"

    # Скачиваем фото в указанный путь
    await message.photo[-1].download(destination_file=image_path)

    # Открываем скачанное изображение
    image = Image.open(image_path).convert("RGB")

    # Классифицируем изображение
    category = classify_image(image)

    # Определяем путь для окончательного сохранения
    save_path = f"images/{'top' if category == 'верх' else 'bottom'}/{filename}"
    image.save(save_path)  # Сохраняем

    await message.answer(f"Фото сохранено в категорию: {category}!")  # Отправляем ответ

@dp.message_handler(lambda message: message.text == "Собрать образ")
async def assemble_outfit(message: types.Message):
    """Выбираем случайный верх и низ, объединяем и отправляем пользователю."""
    tops = os.listdir("images/top")
    bottoms = os.listdir("images/bottom")

    if not tops or not bottoms:
        await message.answer("Недостаточно фото для сборки образа! Добавь больше вещей.")
        return

    top_image = Image.open(f"images/top/{random.choice(tops)}")
    bottom_image = Image.open(f"images/bottom/{random.choice(bottoms)}")

    # Объединяем изображения
    width = max(top_image.width, bottom_image.width)
    new_height = top_image.height + bottom_image.height
    outfit = Image.new("RGB", (width, new_height), (255, 255, 255))
    outfit.paste(top_image, (0, 0))
    outfit.paste(bottom_image, (0, top_image.height))

    # Сохраняем и отправляем
    output_path = "outfit.jpg"
    outfit.save(output_path)
    with open(output_path, "rb") as photo:
        await message.answer_photo(photo, caption="Твой случайный образ!")

@dp.message_handler(lambda message: message.text == "Посмотреть гардероб")
async def view_closet(message: types.Message):
    """Отправляет пользователю все фото одежды."""
    top_files = glob.glob("images/top/*.jpg")
    bottom_files = glob.glob("images/bottom/*.jpg")

    if not top_files and not bottom_files:
        await message.answer("Гардероб пуст! Добавь одежду.")
        return

    await message.answer("Твой гардероб:")

    # Отправляем все фото верха
    if top_files:
        await message.answer("👕 Верхняя одежда:")
        for file in top_files:
            with open(file, "rb") as photo:
                await message.answer_photo(photo)

    # Отправляем все фото низа
    if bottom_files:
        await message.answer("👖 Нижняя одежда:")
        for file in bottom_files:
            with open(file, "rb") as photo:
                await message.answer_photo(photo)

@dp.message_handler(lambda message: message.text == "Очистить гардероб")
async def clear_closet(message: types.Message):
    """Удаляет все сохранённые вещи."""
    for folder in ["images/top", "images/bottom"]:
        files = glob.glob(f"{folder}/*.jpg")
        for file in files:
            os.remove(file)

    await message.answer("Гардероб очищен! Теперь можно загружать новую одежду.")

async def on_startup(_):
    await set_commands()

executor.start_polling(dp, skip_updates=True)
