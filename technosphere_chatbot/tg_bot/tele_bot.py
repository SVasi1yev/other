import telebot
import bot_config
import time
from collections import deque


def setup_message(user_id, message_id):
    global users_dict

    if user_id not in users_dict:
        if len(users_dict) > bot_config.MAX_USERS:
            users_dict = {}
        users_dict[user_id] = bot_config.User()
    else:
        try:
            bot.edit_message_reply_markup(chat_id=user_id, message_id=message_id)
        except telebot.apihelper.ApiTelegramException:
            pass


users_dict = {}

bot = telebot.TeleBot(token=bot_config.TG_TOKEN)
default_markup = telebot.types.InlineKeyboardMarkup()
default_markup.add(telebot.types.InlineKeyboardButton(text='Change character', callback_data='/change_model'))
default_markup.add(telebot.types.InlineKeyboardButton(text='Start dialog', callback_data='/config_dialog'))


@bot.message_handler(commands=['start'])
def start_command_handler(message):
    global users_dict

    user_id = message.from_user.id
    setup_message(user_id, message.message_id - 1)

    user = users_dict[user_id]

    response = f'Hi, I\'m a bot that can chat with you in the style of one of your favorite characters. \n' \
               f'Now {user.model_name} is selected.'
    bot.send_message(chat_id=user_id, text=response, reply_markup=default_markup)


@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    global users_dict

    user_id = message.from_user.id
    setup_message(user_id, message.message_id - 1)

    user = users_dict[user_id]
    if user.run_dialog:
        user.run_dialog = False
        return
    text = message.text

    user.add_context(text)
    response = user.model.get_response(context=user.context)
    if response == '':
        response = ' '
    bot.send_message(chat_id=user_id, text=response, reply_markup=default_markup)


@bot.callback_query_handler(func=lambda call: call.data == '/start')
def start_handler(call):
    global users_dict

    user_id = call.message.chat.id
    setup_message(user_id, call.message.message_id)

    user = users_dict[user_id]

    response = f'Hi, I\'m a bot that can chat with you in the style of one of your favorite characters. \n' \
               f'Now {user.model_name} is selected.'
    bot.send_message(chat_id=user_id, text=response, reply_markup=default_markup)


@bot.callback_query_handler(func=lambda call: call.data in ['/change_model', '/homer_medium',
                                                            '/homer_large', '/bart_medium',
                                                            'homer_small', '/bart_small'])
def change_model_handler(call):
    global users_dict

    user_id = call.message.chat.id
    setup_message(user_id, call.message.message_id)

    user = users_dict[user_id]

    if call.data == '/change_model':
        response = 'Select a character.'
        markup = telebot.types.InlineKeyboardMarkup()
        markup.add(telebot.types.InlineKeyboardButton(text='Homer_medium', callback_data='/homer_medium'))
        markup.add(telebot.types.InlineKeyboardButton(text='Homer_large', callback_data='/homer_large'))
        markup.add(telebot.types.InlineKeyboardButton(text='Bart_medium', callback_data='/bart_medium'))
        # markup.add(telebot.types.InlineKeyboardButton(text='Homer_small', callback_data='/homer_small'))
        # markup.add(telebot.types.InlineKeyboardButton(text='Bart_small', callback_data='/bart_small'))
        markup.add(telebot.types.InlineKeyboardButton(text='<- Back', callback_data='/start'))
        bot.send_message(chat_id=user_id, text=response, reply_markup=markup)
    else:
        if call.data == '/homer_medium':
            user.set_model('Homer_medium')
        elif call.data == '/homer_large':
            user.set_model('Homer_large')
        elif call.data == '/bart_medium':
            user.set_model('Bart_medium')
        elif call.data == '/homer_small':
            user.set_model('Homer_small')
        elif call.data == '/bart_small':
            user.set_model('Bart_small')
        user.context = deque()
        response = f'Сhanged the character to {user.model_name}.'
        bot.send_message(chat_id=user_id, text=response, reply_markup=default_markup)


@bot.callback_query_handler(
    func=lambda call: call.data in ['/config_dialog',
                                    '/homer_medium_01', '/homer_medium_02',
                                    '/homer_large_01', '/homer_large_02',
                                    '/bart_medium_01', '/bart_medium_02',
                                    '/homer_small_01', '/homer_small_02',
                                    '/bart_small_01', '/bart_small_02']
)
def config_dialog_handler(call):
    global users_dict

    user_id = call.message.chat.id
    setup_message(user_id, call.message.message_id)

    user = users_dict[user_id]

    if call.data == '/config_dialog':
        response = 'Select first character.'
        markup = telebot.types.InlineKeyboardMarkup()
        markup.add(telebot.types.InlineKeyboardButton(text='Homer_medium', callback_data='/homer_medium_01'))
        markup.add(telebot.types.InlineKeyboardButton(text='Homer_large', callback_data='/homer_large_01'))
        markup.add(telebot.types.InlineKeyboardButton(text='Bart_medium', callback_data='/bart_medium_01'))
        # markup.add(telebot.types.InlineKeyboardButton(text='Homer_small', callback_data='/homer_small_01'))
        # markup.add(telebot.types.InlineKeyboardButton(text='Bart_small', callback_data='/bart_small_01'))
        markup.add(telebot.types.InlineKeyboardButton(text='<- Back', callback_data='/start'))
        bot.send_message(chat_id=user_id, text=response, reply_markup=markup)
    elif call.data == '/homer_medium_01' or call.data == '/homer_large_01'\
            or call.data == '/bart_medium_01' or call.data == '/homer_small_01'\
            or call.data == '/bart_small_01':
        markup = telebot.types.InlineKeyboardMarkup()
        markup.add(telebot.types.InlineKeyboardButton(text='Homer_medium', callback_data='/homer_medium_02'))
        markup.add(telebot.types.InlineKeyboardButton(text='Homer_large', callback_data='/homer_large_02'))
        markup.add(telebot.types.InlineKeyboardButton(text='Bart_medium', callback_data='/bart_medium_02'))
        # markup.add(telebot.types.InlineKeyboardButton(text='Homer_small', callback_data='/homer_small_02'))
        # markup.add(telebot.types.InlineKeyboardButton(text='Bart_small', callback_data='/bart_small_02'))
        markup.add(telebot.types.InlineKeyboardButton(text='<- Back', callback_data='/config_dialog'))
        if call.data == '/homer_medium_01':
            user.dialog_models[0] = 'Homer_medium'
        elif call.data == '/homer_large_01':
            user.dialog_models[0] = 'Homer_large'
        elif call.data == '/bart_medium_01':
            user.dialog_models[0] = 'Bart_medium'
        elif call.data == '/homer_small_01':
            user.dialog_models[0] = 'Homer_small'
        elif call.data == '/bart_small_01':
            user.dialog_models[0] = 'Bart_small'
        response = f'First character is {user.dialog_models[0]}. Select second character.'
        bot.send_message(chat_id=user_id, text=response, reply_markup=markup)
    elif call.data == '/homer_medium_02' or call.data == '/homer_large_02'\
            or call.data == '/bart_medium_02' or call.data == '/homer_small_02'\
            or call.data == '/bart_small_02':
        markup = telebot.types.InlineKeyboardMarkup()
        markup.add(telebot.types.InlineKeyboardButton(text='Start', callback_data='/start_dialog'))
        markup.add(telebot.types.InlineKeyboardButton(
            text='<- Back', callback_data=f'/{user.dialog_models[0].lower()}_01')
        )
        if call.data == '/homer_medium_02':
            user.dialog_models[1] = 'Homer_medium'
        elif call.data == '/homer_large_02':
            user.dialog_models[1] = 'Homer_large'
        elif call.data == '/bart_medium_02':
            user.dialog_models[1] = 'Bart_medium'
        elif call.data == '/homer_small_02':
            user.dialog_models[1] = 'Homer_small'
        elif call.data == '/bart_small_02':
            user.dialog_models[1] = 'Bart_small'
        user.wait_length = True
        response = f'Second character is {user.dialog_models[1]}. \n' \
                   'You can start dialog.'
        bot.send_message(chat_id=user_id, text=response, reply_markup=markup)


@bot.callback_query_handler(func=lambda call: call.data == '/start_dialog')
def start_handler(call):
    global users_dict

    user_id = call.message.chat.id
    setup_message(user_id, call.message.message_id)

    response = 'Enter something to stop dialog.'
    bot.send_message(chat_id=user_id, text=response)
    time.sleep(1)

    user = users_dict[user_id]

    START_REPLICA = 'Hi, how are you?'

    model_names = [user.dialog_models[0], user.dialog_models[1]]
    models = [bot_config.models_dict[model_names[0]], bot_config.models_dict[model_names[1]]]
    dialog_contexts = [deque(), deque()]
    dialog_contexts[0].append(models[0].tokenizer.encode(START_REPLICA + models[0].tokenizer.eos_token, return_tensors='pt'))
    dialog_contexts[1].append(models[1].tokenizer.encode(START_REPLICA + models[1].tokenizer.eos_token, return_tensors='pt'))

    response = model_names[0] + ' >> ' + START_REPLICA
    bot.send_message(chat_id=user_id, text=response)
    time.sleep(1)

    user.run_dialog = True
    i = 1
    while user.run_dialog:
        response = models[i%2].get_response(dialog_contexts[i%2])
        dialog_contexts[(i+1) % 2].append(
            models[(i+1) % 2].tokenizer.encode(
                response + models[(i+1) % 2].tokenizer.eos_token, return_tensors='pt'
            )
        )
        while len(dialog_contexts[i%2]) > models[i%2].context_len:
            dialog_contexts[i%2].popleft()
        response = model_names[i%2] + ' >> ' + response
        if user.run_dialog:
            bot.send_message(chat_id=user_id, text=response)
        else:
            break
        time.sleep(1)
        i += 1
        if i == bot_config.MAX_DIALOG_LEN:
            break
    response = 'End of dialog.'
    bot.send_message(chat_id=user_id, text=response, reply_markup=default_markup)


bot.polling(none_stop=True, interval=0)
