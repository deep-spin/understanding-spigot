import time

def printable_time():
    return time.strftime("%x %X", time.gmtime())

def write_line_to_file(filename, *text):
    text = ' '.join(map(str, text))
    print(text)
    with open(filename, 'a', encoding="utf-8") as out:
        out.write(text)
        out.write('\n')
