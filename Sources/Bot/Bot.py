import os
import shutil
import threading
from functools import wraps

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, ChatAction, InputMediaPhoto
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, MessageHandler, Filters, CallbackContext

from Sources.Face_Master import get_smeared_faces, final_swap_and_clear
from Sources.Util.Configs import get_telegram_token
from Sources.Util.State import TaskState
from defenitions import ROOT_DIR

home = ROOT_DIR


def send_upload_photo(func):
    """Sends typing action while processing func command."""

    @wraps(func)
    def command_func(context, update, *args, **kwargs):
        try:
            if isinstance(update, int):
                context.updater.bot.send_chat_action(chat_id=update, action=ChatAction.UPLOAD_PHOTO)
            else:
                context.updater.bot.send_chat_action(chat_id=update.effective_message.chat_id, action=ChatAction.UPLOAD_PHOTO)
        except Exception:
            pass
        finally:
            return func(context, update, *args, **kwargs)

    return command_func


class SwapBot:
    chats_ids_with_photo_data = dict()
    chats_ids_with_states = dict()  # id: index
    chats_ids_with_choices = set()

    photos_files = list()
    session_state = TaskState.WAITING_FOR_PHOTOS
    target_photo_filename = ""

    admin_id = -1

    users_count = 123123123123
    processing = False

    def __init__(self):
        self.updater = Updater(get_telegram_token(), use_context=True)

        def start_message(update, context):
            if self.session_state == TaskState.WAITING_FOR_PHOTOS:
                self.updater.bot.send_message(update.message.chat_id, "Добрый день! Жду Ваших фото")
            elif self.session_state == TaskState.WAITING_FOR_CHOICES:
                self.send_gallery(update.message.chat_id, "Выберите то фото, на котором Вы получились лучше всего")

        def process_photos(update, context):
            if len(update.message.photo) != 0:
                file_name = "{0}/photos/{1}.jpg".format(home, update.message.photo[-1].file_id)
                self.updater.bot.getFile(update.message.photo[-1].file_id).download(file_name)

                self.photos_files.append(file_name)

        def photos_sent_message(update, context):
            if self.admin_id == -1:
                self.admin_id = update.message.chat_id

            if update.message.chat_id == self.admin_id:
                self.session_state = TaskState.WAITING_FOR_CHOICES

                self.send_gallery(update.message.chat_id, "Выберите целевую фотографию:")

        def make_message(update, context):
            if not self.processing:
                threaded = threading.Thread(target=self.swap_all_faces)
                threaded.start()

        self.updater.dispatcher.add_handler(CommandHandler("start", start_message))
        self.updater.dispatcher.add_handler(CommandHandler("photos_sent", photos_sent_message))
        self.updater.dispatcher.add_handler(CommandHandler("make", make_message))
        self.updater.dispatcher.add_handler(MessageHandler(Filters.photo, process_photos))
        self.updater.dispatcher.add_handler(CallbackQueryHandler(self.button))

        self.updater.start_polling()

    def send_gallery(self, chat_id: int, reply_text: str):
        keyboard = [
            [
                InlineKeyboardButton("➡️", callback_data='p>'),
            ],
            [InlineKeyboardButton("✅", callback_data='pv')],
        ]

        reply_markup = InlineKeyboardMarkup(keyboard)

        self.chats_ids_with_states[chat_id] = 0
        self.updater.bot.send_message(chat_id, text=reply_text)
        self.updater.bot.send_photo(chat_id, photo=open(self.photos_files[0], 'rb'), reply_markup=reply_markup)

    def send_faces_gallery(self, chat_id: int):
        keyboard = [
            [
                InlineKeyboardButton("➡️", callback_data='f>')
            ],
            [InlineKeyboardButton("✅", callback_data='fv')],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        self.chats_ids_with_states[chat_id] = 0

        self.updater.bot.send_message(chat_id, text="Выберите ваше лицо")
        self.updater.bot.send_photo(chat_id, photo=open(f"{home}/faces/{chat_id}_{str(0)}.jpg", 'rb'),
                                    reply_markup=reply_markup)

    @send_upload_photo
    def button(self, update: Update, context: CallbackContext) -> None:
        query = update.callback_query
        query.answer()

        data = query.data
        index = self.chats_ids_with_states[query.message.chat_id]

        if data[0] == 'p':
            keyboard = [
                [
                    InlineKeyboardButton("➡️", callback_data='p>'),
                    InlineKeyboardButton("⬅️", callback_data='p<'),
                ],
                [InlineKeyboardButton("✅", callback_data='pv')],
            ]

            reply_markup = InlineKeyboardMarkup(keyboard)

            if data == 'p>':
                if index + 1 < len(self.photos_files):
                    index += 1

                    if index == len(self.photos_files) - 1:
                        del keyboard[0][0]

                        reply_markup = InlineKeyboardMarkup(keyboard)

                    self.chats_ids_with_states[query.message.chat_id] += 1
                    query.edit_message_media(media=InputMediaPhoto(open(self.photos_files[index], 'rb')), reply_markup=reply_markup)
            elif data == 'p<':
                if index > 0:
                    index -= 1

                    if index == 0:
                        keyboard[0].pop()
                        reply_markup = InlineKeyboardMarkup(keyboard)

                    self.chats_ids_with_states[query.message.chat_id] -= 1
                    query.edit_message_media(media=InputMediaPhoto(open(self.photos_files[index], 'rb')), reply_markup=reply_markup)
            else:
                if query.message.chat_id == self.admin_id and self.target_photo_filename == "":
                    self.target_photo_filename = self.photos_files[index]
                    self.updater.bot.delete_message(query.message.chat_id, query.message.message_id)

                    self.send_gallery(query.message.chat_id, "Выберите то фото, на котором Вы получились лучше всего")
                else:
                    self.updater.bot.delete_message(query.message.chat_id, query.message.message_id)
                    if not self.processing:
                        threaded = threading.Thread(target=self.mark_photo_for_user, args=(query.message.chat_id, self.photos_files[index]))
                        threaded.start()
        else:
            keyboard = [
                [
                    InlineKeyboardButton("➡️", callback_data='f>'),
                    InlineKeyboardButton("⬅️", callback_data='f<'),
                ],
                [InlineKeyboardButton("✅", callback_data='fv')],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            if data == 'f>':
                if index + 1 < self.chats_ids_with_photo_data[query.message.chat_id][0]:
                    index += 1

                    if index == self.chats_ids_with_photo_data[query.message.chat_id][0] - 1:
                        del keyboard[0][0]
                        reply_markup = InlineKeyboardMarkup(keyboard)

                    self.chats_ids_with_states[query.message.chat_id] += 1
                    query.edit_message_media(
                        media=InputMediaPhoto(open(f"{home}/faces/{query.message.chat_id}_{str(index)}.jpg", 'rb')), reply_markup=reply_markup)
            elif data == 'f<':
                if index > 0:
                    index -= 1

                    if index == 0:
                        keyboard[0].pop()
                        reply_markup = InlineKeyboardMarkup(keyboard)

                    query.edit_message_media(
                        media=InputMediaPhoto(open(f"{home}/faces/{query.message.chat_id}_{str(index)}.jpg", 'rb')), reply_markup=reply_markup)
            else:
                self.chats_ids_with_photo_data[query.message.chat_id] = (
                    index, f"{query.message.chat_id}_{str(index)}.jpg",
                    self.chats_ids_with_photo_data[query.message.chat_id][1],
                    self.chats_ids_with_photo_data[query.message.chat_id][2],
                    self.chats_ids_with_photo_data[query.message.chat_id][3])

                self.chats_ids_with_choices.add(query.message.chat_id)
                self.updater.bot.delete_message(query.message.chat_id, query.message.message_id)
                self.updater.bot.send_message(query.message.chat_id, "Отлично! Как только все пользователи выберут "
                                                                     "свои фото "
                                                                     "- я пришлю результат!")

                if len(self.chats_ids_with_choices) == self.users_count:
                    self.swap_all_faces()

    def clear_all(self):
        shutil.rmtree(f"{home}/photos")
        shutil.rmtree(f"{home}/faces")
        shutil.rmtree(f"{home}/results")

        os.mkdir(f"{home}/photos")
        os.mkdir(f"{home}/faces")
        os.mkdir(f"{home}/results")

        self.processing = False

    def swap_all_faces(self):
        self.processing = True
        result_file = final_swap_and_clear(self.target_photo_filename,
                                           list(self.chats_ids_with_photo_data.values()))

        for chat in self.chats_ids_with_choices:
            self.updater.bot.send_photo(chat_id=chat, photo=open(result_file))

        self.clear_all()

    @send_upload_photo
    def mark_photo_for_user(self, chat_id, file_name):
        self.chats_ids_with_photo_data[chat_id] = get_smeared_faces(chat_id, file_name)
        self.users_count = min(self.users_count, self.chats_ids_with_photo_data[chat_id][0])
        self.send_faces_gallery(chat_id)
