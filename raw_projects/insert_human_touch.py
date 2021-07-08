# +
from random import choice
import datetime

def get_today_dayofweek():
    #daynames = ("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday")
    daynames = ("Monday","Tuesday","Wednesday","Thursday","Friday","weekend","Sunday")
    return daynames[datetime.datetime.today().weekday()]

def gen_regards():
    options = ['Kind regards',
               'Best regards',
               'Regards',
               'Best wishes',
               'All the best',
               'Best'
              ]
    return choice(options)
    
def gen_haveaniceday():
    options = ['Have a great day.',
                f'Have a great {get_today_dayofweek()}.',
               'I hope that you have a productive day.',
               f'I hope that you have a productive {get_today_dayofweek()}.',
               'I hope that you have a great day.',
               'Have a great day',
               f'I hope that {get_today_dayofweek()} treats you well.',
               'Have a wonderful day.',
               f'Have a wonderful {get_today_dayofweek()}.',
               'I hope that you have a wonderful day.',
               'I hope that your day goes well today.',
               'Have a wonderful day further.',
               'I hope that you have a stress-free day.',
               'I hope that everything goes well today.',
               'I hope that your day is productive.'
              ]
    return choice(options)
    
def gen_attachment_msg(attachment_desc):
    options = [ f'Please see attached the {attachment_desc}.',
                f'The {attachment_desc} are attached.',
                f'Please see attached the {attachment_desc}.',
                f'Please see attached the {attachment_desc} that you requested.',
                f'Please see attached the {attachment_desc} that you asked for.',
                f"I've attached the {attachment_desc}.",
                f'Please see the {attachment_desc} attached to this email.',
                f'Here are the {attachment_desc}.',
                f'Here are the {attachment_desc} that you requested.',
                f'The {attachment_desc}.',
                f'Freshly made {attachment_desc}.'
              ]
    return choice(options)


# -

for i in range(20):
    assemble_message = 'Dear Albert\n\n' + gen_attachment_msg('FTB numbers') + '\n\n' + gen_haveaniceday() + '\n\n' + gen_regards() + '\nJoe'
    print(assemble_message)                   
    print('\n\n\n\n\n')


