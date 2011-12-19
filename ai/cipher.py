import string


def dorotate(text, i):
    out = []
    for c in text:
        v = ord(c)
        if (97 <= v <= 122):
            v -= i
            if v < 97:
                v = 122 - (96 - v)
        out += [chr(v)]
    return "".join(out)



if __name__ == "__main__":
    sentence = "esp qtcde nzyqpcpynp zy esp ezatn zq lcetqtntlw tyepwwtrpynp hld spwo le olcexzfes nzwwprp ty estd jpclc"

    for i in range(1, 26):
        rot = dorotate(sentence, i)
        print i, rot
    # dorotate(sentence, 10)
