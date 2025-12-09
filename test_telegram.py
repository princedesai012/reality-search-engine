import asyncio
from telegram import Bot

# Yahan apna details daalo
BOT_TOKEN = "8404374297:AAEKS20itGajFn5-GjzopFvFyCyVsh-Zook"  # BotFather wala token
CHAT_ID = "6285318225"      # UserInfoBot wala number

async def send_test_message():
    try:
        bot = Bot(token=BOT_TOKEN)
        print("Sending test message...")
        await bot.send_message(chat_id=CHAT_ID, text="🚨 Test Alert: System is Online!")
        print("✅ Message sent successfully!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(send_test_message())