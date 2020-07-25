import numpy as np


def textCut(text):
    returnText = [s.strip('。|，|、|)|:|{|}|“|”|□|.|(|（|）|?|？|·|\r|\n') for s in text]
    stopWords = '而|何|乎|乃|其|且|若|所|为|焉|也|以|因|于|与|则|者|之|不|自|得|一|来|去|经|\
        无|可|是|已|此|的|上|中|兮|三|有|长|见|在|里|未|送|作|事|处|多|谁|向|五|二|后|看|大|\
        首|复|外|过|州|能|莫|下|子|出|千|百|入|万|应|还|方|名|似|发|故|犹|边|九|分|初|坐|期|\
        当|怀|意|四|起|几|间|到|中|关|中|然|作|随|亦|非|终|相|生|如|今|前|将|开|欲|尽|更|重|\
        同|从|王|平|十|安|连'

    returnText = [s for s in returnText if s not in stopWords]
    return list(filter(None, returnText))


def createPeotryTexts():
    fr = open('peotry.txt', encoding='gb18030')
    peotryTexts = []
    peotryItem = []
    textItem = ''
    for line in fr.readlines():
        if len(line) == 1:
            continue
        if line[0] == '第' and line[len(line) - 2] == '卷':
            continue
        if line.find('【') > 0:
            if len(textItem) > 0:
                if len(peotryItem) == 1:
                    peotryItem.append(' ')
                peotryItem.append(textItem)
                peotryTexts.append(peotryItem)
                textItem = ''
            if line.find('（') >= 0 and line.find('）') >= 0:
                line = line.replace(line[line.find('（'):line.find('）') + 1], '')
            title = line[line.find('【') + 1:line.find('】')]
            peotryItem = []
            peotryItem.append(title)
            if line.find('】') + 2 < len(line):
                peotryItem.append(line[line.find('】') + 1:-1].strip(' ').strip('\n'))
            continue
        if len(peotryItem) == 1 and len(line) < 7:
            peotryItem.append(line.strip().strip('\n'))
            continue
        if line.find('（') >= 0:
            if line.find('）') >= 0:
                line = line.replace(line[line.find('（'):line.find('）') + 1], '')
            else:
                line = line.replace(line[line.find('（'):], '')
        elif line.find('）') >= 0:
            line = line.replace(line[:line.find('）') + 1], '')
        if line.find('(') >= 0:
            if line.find(')') >= 0:
                line = line.replace(line[line.find('('):line.find(')') + 1], '')
            else:
                line = line.replace(line[line.find('('):], '')
        elif line.find(')') >= 0:
            line = line.replace(line[:line.find(')') + 1], '')
        if line.find('--') >= 0:
            line = line.replace(line[line.find('--'):], '')
        if line.rfind('。') > 0 and line.rfind('。') + 3 == len(line):
            line = line[:line.rfind('。') + 1]
        if len(line) <= 2:
            continue
        textItem += line.strip().strip('\n')
        line = fr.readline()
    peotryItem.append(textItem)
    peotryTexts.append(peotryItem)
    return peotryTexts


def createCharDict(peotryTexts):
    charDict = {}
    for peotryItem in peotryTexts:
        strList = textCut(peotryItem[2])
        for charItem in strList:
            if charItem not in charDict:
                charDict[charItem] = 1
            else:
                charDict[charItem] += 1
    lenPeotryTexts = len(peotryTexts)
    for item in charDict.keys():
        charDict[item] = round(charDict[item] / lenPeotryTexts, 4)
    charDict = sorted(charDict.items(), key=lambda d: d[1], reverse=True)
    return charDict


if __name__ == "__main__":
    # text = '云横秦岭家何在，雪拥蓝关马不前。'
    # print(textCut(text))
    peotryTexts = createPeotryTexts()
    charDict = createCharDict(peotryTexts)
    print(charDict[:148])

    # import re
    # res = ''.join(re.findall(r'[\u4E00-\u9FA5]+', '我是s 汉字'))
    # print(res)  # <re.Match object; span=(0, 2), match='我是'>
