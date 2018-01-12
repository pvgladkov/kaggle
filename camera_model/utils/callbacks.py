import io
import telegram

from keras.callbacks import Callback
from keras.callbacks import (ModelCheckpoint, EarlyStopping)
from kaggle.utils import create_logger

logger = create_logger('telegram monitor')


class TelegramMonitor(Callback):
    """This monitor send messages to a Telegram chat ID with a bot.
    """

    def __init__(self, api_token, chat_id, model_name):

        super(TelegramMonitor, self).__init__()

        self.bot = telegram.Bot(token=api_token)
        self.chat_id = chat_id
        self.model_name = model_name

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}
        nice_logs = ' | '.join(["{} = {:.6f}".format(k, v) for k, v in logs.items()])

        # Write and send message
        message = "{}: Epoch {}/{} is done. {}"
        self.notify(message.format(self.model_name, epoch + 1, self.params['epochs'], nice_logs))

    def on_train_end(self, logs=None):
        self.notify('{}: Finish'.format(self.model_name))

    def notify(self, message, parse_mode=None):
        try:
            ret = self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode=parse_mode)
            return ret
        except Exception as e:
            logger.exception(e)


def init_callbacks(w_file, m_name, v, telegram_bot_api_key, chat_it):
    checkpoint = ModelCheckpoint(w_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    _name = '{}-{}'.format(m_name, v)
    telegram = TelegramMonitor(api_token=telegram_bot_api_key, chat_id=chat_it, model_name=_name)
    return [checkpoint, telegram]