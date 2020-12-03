from Sources.Bot.Bot import SwapBot

if __name__ == '__main__':
    bot = SwapBot()
    bot.updater.start_polling()