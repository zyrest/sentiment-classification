from xml.dom.minidom import parse
import xml.dom.minidom


def handle_file(file_name, category):
    f_train = open('weibo/weibo.train.txt', 'a', encoding='utf-8')
    f_val = open('weibo/weibo.val.txt', 'a', encoding='utf-8')
    f_test = open('weibo/weibo.test.txt', 'a', encoding='utf-8')

    # 使用minidom解析器打开 XML 文档
    DOMTree = xml.dom.minidom.parse(file_name)
    collection = DOMTree.documentElement

    # 在集合中获取所有
    reviews = collection.getElementsByTagName("review")

    # 详细信息
    count = 0
    for review in reviews:
        content = str(review.childNodes[0].data).replace('\n', '').replace('\t', '').replace(' ', '')
        if count < 4000:
            f_train.write(category + '\t' + content + '\n')
        elif count < 4500:
            f_val.write(category + '\t' + content + '\n')
        elif count < 5000:
            f_test.write(category + '\t' + content + '\n')
        count += 1


if __name__ == '__main__':
    handle_file('sample.negative.txt', 'neg')
    handle_file('sample.positive.txt', 'pos')
