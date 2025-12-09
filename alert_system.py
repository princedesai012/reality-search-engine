import os
import logging
from telegram import Bot
from telegram.error import TelegramError
import asyncio

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

async def send_telegram_alert(image_path: str, label: str, confidence: float) -> bool:
    """
    Send an alert with the detected image to Telegram.
    
    Args:
        image_path: Path to the image file
        label: Detected label (e.g., 'person', 'suspicious activity')
        confidence: Detection confidence (0-1)
        
    Returns:
        bool: True if alert was sent successfully, False otherwise
    """
    # Get credentials from environment variables
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        logger.error("Telegram credentials not found in environment variables")
        return False
    
    try:
        bot = Bot(token=bot_token)
        confidence_pct = f"{confidence * 100:.1f}%"
        caption = f"⚠️ Alert: {label} detected ({confidence_pct})"
        
        with open(image_path, 'rb') as photo:
            await bot.send_photo(
                chat_id=chat_id,
                photo=photo,
                caption=caption,
                parse_mode='Markdown'
            )
        
        logger.info(f"Alert sent: {label} ({confidence_pct})")
        return True
        
    except TelegramError as e:
        logger.error(f"Failed to send Telegram alert: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error sending alert: {e}")
        return False

def send_alert(image_path: str, label: str, confidence: float) -> bool:
    """
    Synchronous wrapper for the async send_telegram_alert function.
    """
    try:
        return asyncio.run(send_telegram_alert(image_path, label, confidence))
    except Exception as e:
        logger.error(f"Error in send_alert: {e}")
        return False