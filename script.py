#!/usr/bin/env python
import nltk, random
from nltk.corpus import stopwords
from nltk import FreqDist, pos_tag
from nltk.classify import NaiveBayesClassifier
from nltk.stem import WordNetLemmatizer
from math import log, floor

def parsing():
    test_file = open("TEST_FILE.txt", 'r').read()
    train_file = open("TRAIN_FILE.txt", 'r').read()
    offset = 0
    #f = open("train.txt", "r").read().split('\n')
    test = test_file.split('\n')
    train = train_file.split('\n')
    for i in xrange(len(train)):
        if i % 4 == 1:
            ans_train.append(train[i].split('\r')[0])
        if i % 4 == 0:
            train[i] = train[i].replace('<e2>', ' ')
            train[i] = train[i].replace('</e2>', ' ')
            train[i] = train[i].replace('<e1>', ' ')
            train[i] = train[i].replace('</e1>', ' ')
            snippet_train.append(train[i].split(' '))
            
    for i in xrange(len(test)):
        snippet_test.append(test[i].split(' '))
        
    for i in xrange(len(snippet_train)):       
        snippet_train[i][0] = snippet_train[i][0].split('\t"')[1]
        snippet_train[i][-1] = snippet_train[i][-1].split('."\r')[0]

    for i in xrange(0, len(snippet_test) - 1):
        for j in xrange(len(snippet_test[i])):
            snippet_test[i][j] = snippet_test[i][j].replace('<e2>', '')
            snippet_test[i][j] = snippet_test[i][j].replace('</e2>', '')
            snippet_test[i][j] = snippet_test[i][j].replace('<e1>', '')
            snippet_test[i][j] = snippet_test[i][j].replace('</e1>', '')
            snippet_test[i][j] = snippet_test[i][j].replace(',\'', '')
            snippet_test[i][j] = snippet_test[i][j].replace('\'', '')
            snippet_test[i][j] = snippet_test[i][j].replace(',', '')
            snippet_test[i][j] = snippet_test[i][j].replace(';', '')
            snippet_test[i][j] = snippet_test[i][j].replace(':', '')
            snippet_test[i][j] = snippet_test[i][j].replace('\"', '')
        snippet_test[i][0] = snippet_test[i][0].split('\t')[1]
        snippet_test[i][-1] = snippet_test[i][-1].split('.\r')[0]
    #print snippet_test[256], len(snippet_test)
    #print snippet_train[17], len(snippet_train)

    test = test_file.split('\t')
    train = train_file.split('\t')
    for i in train:
        indexA = i.find('<e1>')
        indexB = i.find('</e1>')
        train_e1.append(i[indexA + 4 : indexB].lower())
        #print train_e1[-1]
        indexA = i.find('<e2>')
        indexB = i.find('</e2>')
        train_e2.append(i[indexA + 4 : indexB].lower())
        
    for i in test:
        indexA = i.find('<e1>')
        indexB = i.find('</e1>')
        test_e1.append(i[indexA + 4 : indexB].lower())
        indexA = i.find('<e2>')
        indexB = i.find('</e2>')
        test_e2.append(i[indexA + 4 : indexB].lower())

    return test_e1[1:], test_e2[1:], train_e1[1:], train_e2[1:], ans_train, snippet_train, snippet_test[:-1]

def output():
    #assert len(ans) == 2717
    f = open("myans2.txt", "w")
    count = 8001
    for i in myans:
        if i == 1:
            f.write(str(count ) + "\tCause-Effect(e1,e2)\n")
        elif i == -1:
            f.write(str(count ) + "\tCause-Effect(e2,e1)\n")
        elif i == 2:
            f.write(str(count ) + "\tInstrument-Agency(e1,e2)\n")
        elif i == -2:
            f.write(str(count ) + "\tInstrument-Agency(e2,e1)\n")
        elif i == 3:
            f.write(str(count ) + "\tProduct-Producer(e1,e2)\n")
        elif i == -3:
            f.write(str(count ) + "\tProduct-Producer(e2,e1)\n")
        elif i == 4:
            f.write(str(count ) + "\tContent-Container(e1,e2)\n")
        elif i == -4:
            f.write(str(count ) + "\tContent-Container(e2,e1)\n")
        elif i == 5:
            f.write(str(count ) + "\tEntity-Origin(e1,e2)\n")
        elif i == -5:
            f.write(str(count ) + "\tEntity-Origin(e2,e1)\n")
        elif i == 6:
            f.write(str(count ) + "\tEntity-Destination(e1,e2)\n")
        elif i == -6:
            f.write(str(count ) + "\tEntity-Destination(e2,e1)\n")
        elif i == 7:
            f.write(str(count ) + "\tComponent-Whole(e1,e2)\n")
        elif i == -7:
            f.write(str(count ) + "\tComponent-Whole(e2,e1)\n")
        elif i == 8:
            f.write(str(count ) + "\tMember-Collection(e1,e2)\n")
        elif i == -8:
            f.write(str(count ) + "\tMember-Collection(e2,e1)\n")
        elif i == 9:
            f.write(str(count ) + "\tMessage-Topic(e1,e2)\n")
        elif i == -9:
            f.write(str(count ) + "\tMessage-Topic(e2,e1)\n")
        else:
            f.write(str(count ) + "\tOther\n")
        count += 1

def preprocessing():
    lemmatizer = WordNetLemmatizer()
    stopword = open("stopwords.txt", 'r').read().split('\n')[:-1]
    #print stopwords
    for i in xrange(len(snippet_train)):
        for j in xrange(len(snippet_train[i])):
            snippet_train[i][j] = lemmatizer.lemmatize(snippet_train[i][j].lower())
        snippet_train[i] = [word for word in snippet_train[i] if word not in stopword]
    #print snippet_test[510]
    #print "with" in s
    for i in xrange(len(snippet_test)):
        for j in xrange(len(snippet_test[i])):
            snippet_test[i][j] = lemmatizer.lemmatize(snippet_test[i][j].lower())
        snippet_test[i] = [word for word in snippet_test[i] if word not in stopword]
    #print snippet_test[510]
    count = 0
    for i in xrange(len(snippet_test)):
        flag1 = 0
        flag2 = 0
        x = test_e1[i]
        y = test_e2[i]
        for j in xrange(len(snippet_test[i])):
            if " " in test_e1[i]:
                x = test_e1[i].split(' ')[0]
            if " " in test_e2[i]:
                y = test_e2[i].split(' ')[0]
            
            if snippet_test[i][j] == lemmatizer.lemmatize(x) and flag1 == 0:
                e1_position.append(j)
                flag1 = 1
                count += 1
            if snippet_test[i][j] == lemmatizer.lemmatize(y) and flag2 == 0:
                e2_position.append(j)
                flag2 = 1
        if flag1 == 0 or flag2 == 0:
            print i, snippet_test[i], lemmatizer.lemmatize(test_e1[i]), lemmatizer.lemmatize(test_e2[i])
    k = 0
    print len(e1_position), len(e2_position),
    for i in xrange(len(ans_train)):
        if ans_train[i] == "Cause-Effect(e1,e2)":
            ans.append(1)
        elif ans_train[i] == "Cause-Effect(e2,e1)":
            ans.append(-1)
        elif ans_train[i] == "Instrument-Agency(e1,e2)":
            ans.append(2)
        elif ans_train[i] == "Instrument-Agency(e2,e1)":
            ans.append(-2)
        elif ans_train[i] == "Product-Producer(e1,e2)":
            ans.append(3)
        elif ans_train[i] == "Product-Producer(e2,e1)":
            ans.append(-3)
        elif ans_train[i] == "Content-Container(e1,e2)":
            ans.append(4)
        elif ans_train[i] == "Content-Container(e2,e1)":
            ans.append(-4)
        elif ans_train[i] == "Entity-Origin(e1,e2)":
            ans.append(5)
        elif ans_train[i] == "Entity-Origin(e2,e1)":
            ans.append(-5)
        elif ans_train[i] == "Entity-Destination(e1,e2)":
           ans.append(6)
        elif ans_train[i] == "Entity-Destination(e2,e1)":
            ans.append(-6)
        elif ans_train[i] == "Component-Whole(e1,e2)":
            ans.append(7)
        elif ans_train[i] == "Component-Whole(e2,e1)":
            ans.append(-7)
        elif ans_train[i] == "Member-Collection(e1,e2)":
            ans.append(8)
        elif ans_train[i] == "Member-Collection(e2,e1)":
            ans.append(-8)
        elif ans_train[i] == "Message-Topic(e1,e2)":
            ans.append(9)
        elif ans_train[i] == "Message-Topic(e2,e1)":
           ans.append(-9)
        else:
            ans.append(0)

def features(x, e1, e2):
    before_e1 = e1
    after_e1 = e1
    before_e2 = e2
    after_e2 = e2
    for i in xrange(len(x)):
        if x[i] == e1:
            if i != 0:
                before_e1 = x[i - 1]
            if i != len(x) - 1:
                after_e1 = x[i + 1]
        if x[i] == e2:
            if i != 0:
                before_e2 = x[i - 1]
            if i != len(x) - 1:
                after_e2 = x[i + 1]

    return {'before_e1':before_e1, 'after_e1':after_e1, 'before_e2':before_e2, 'after_e2':after_e2}

def method1():
    #labeled_data = [(snippet_train[i], str(ans[i])) for i in xrange(len(snippet_train))]
    #random.shuffle(labeled_data)
    featuresets = [(features(snippet_train[i], train_e1[i], train_e2[i]), str(ans[i])) for i in xrange(len(snippet_train))]
    classifier = nltk.NaiveBayesClassifier.train(featuresets)
    for i in xrange(len(snippet_test)):
        myans.append(int(classifier.classify(features(snippet_test[i], test_e1[i], test_e2[i]))))

def method2():
    for i in xrange(len(snippet_test)):
        value = 0
        index = 0
        tmp = []
        for k in xrange(len(snippet_train)):
            count = 0
            for j in xrange(len(snippet_test[i])):
                if snippet_test[i][j] in snippet_train[k]:
                    count += 1
            if count > value:
                value = count
                index = k
        #print value, index
        myans.append(ans[index])
    #print myans

def method3():
   # cause_effect = ['caused', 'from', 'cause', 'induced', 'due', 'triggered', 'produced', 'resulted', 'source', 'after', 'achieved', 'originate']
    cause_effect = ['caused', 'cause', 'induced', 'damage', 'resulted', 'triggered', 'result', 'source', 'after', 'achieved', 'originate']
    instrument_agent = ['with', 'using', 'use', 'used', 'build', 'took']
    product_producer = ['factory', 'company', 'working', 'constructed', 'provided', 'creating', 'product', 'make', 'created']
    content_container = ['in', 'inside', 'enclosed', 'encloses', 'filled']
    entity_origin = ['from', 'made', 'derived', 'origin']
    entity_destination = ['into', 'arrived', 'to']
    component_whole = ['part']
    member_collection = ['multiple', 'several', 'joined', 'litter', 'squad', 'herd', 'member', 'cluster']
    message_topic = ['about', 'book', 'news', 'theory', 'defines', 'focus', 'report', 'mention', 'article', 'magazine', 'aim', 'topic', 'chapter', 'theme', 'speech', 'advertised', 'documents', 'publication', 'thesis']
    #passive = ['by', 'from', '', 'with', 'by', 'use', 'from', 'using', 'took', 'used', 'e1e2', 'is', 'wa', 'are', 'by', 'from', 'of', 'e1e2', 'wa', 'are', 'were', 'is', 'e2e1', 'stick', 'e1e2', 'e1e2', 'wa', 'of', 'e1e2', 'in', 'member', 'e2e1', 'are', 'wa', 'been']
    passive_ce = ['by', 'from']
    passive_ia = ['with', 'by', 'use', 'from', 'using', 'took', 'used']
    passive_pp = ['is', 'wa', 'are', 'were', 'by', 'from', 'of']
    passive_cc = ['wa', 'are', 'were', 'is']
    passive_eo = ['stick']
    passive_ed = ['all e1e2']
    passive_cw = ['is', 'wa', 'are', 'were', 'of']
    passive_mc = ['in', 'member']
    passive_mt = ['been', 'is', 'wa', 'are']
    
    method2()
    print 'method2 finish'
    for i in xrange(len(snippet_test)):
        fill = 0
        if i == 4:
            print snippet_test[i]
        for j in xrange(e1_position[i], e2_position[i]):
            if snippet_test[i][j] in cause_effect:
                if i == 4:
                    print "here"
                #myans.append(1)
                myans[i] = 1
                break
            if snippet_test[i][j] in instrument_agent:
                #myans.append(2)
                myans[i] = 2
                break
            if snippet_test[i][j] in product_producer:
                #myans.append(-3)
                myans[i] = -3
                break
            if snippet_test[i][j] in content_container:
                #myans.append(-4)
                myans[i] = -4
                break
            if snippet_test[i][j] in entity_origin:
                #myans.append(5)
                myans[i] = 5
                break
            if snippet_test[i][j] in entity_destination:
                #myans.append(6)
                myans[i] = 6
                break
            if snippet_test[i][j] in component_whole:
                #myans.append(-7)
                myans[i] = -7
                break
            if snippet_test[i][j] in member_collection:
                #myans.append(-8)
                myans[i] = -8
                break
            if snippet_test[i][j] in message_topic:
                #myans.append(9)
                myans[i] = 9
                break
        #if len(myans) == i:
        #    myans.append(0)
            
    for i in xrange(len(snippet_test)):
        if myans[i] != 0:
            for k in xrange(e1_position[i], e2_position[i]):
                if myans[i] == 1 and snippet_test[i][k] in passive_ce:
                    myans[i] *= -1
                    #print "1"
                if myans[i] == 2 and snippet_test[i][k] in passive_ia:
                    myans[i] *= -1
                    #print "2"
                if myans[i] == 3 and snippet_test[i][k] in passive_pp:
                    myans[i] *= -1
                    #print "3"
                if myans[i] == 4 and snippet_test[i][k] in passive_cc:
                    myans[i] *= -1
                    #print "4"
                if myans[i] == 5 and e2_position[i] - e1_position[i] == 1:
                    myans[i] *= -1
                    #print "5"
                    break
                if myans[i] == 7 and snippet_test[i][k] in passive_cw:
                    myans[i] *= -1
                    #print "7"
                if myans[i] == 8 and snippet_test[i][k] in passive_mc:
                    myans[i] *= -1
                    #print "8"
                if myans[i] == 9 and snippet_test[i][k] in passive_mt:
                    myans[i] *= -1
                    #print "9"
    #method2()
    print len(myans)
    
def method4():
    dic = {}
    for i in xrange(len(snippet_train)):
        for j in xrange(len(snippet_train[i])):
            if snippet_train[i][j] in dic:
                dic[snippet_train[i][j]][0] += ans[i]
                dic[snippet_train[i][j]][1] += 1
            else:
                dic[snippet_train[i][j]] = [ans[i], 1, 0]

    for i in xrange(len(snippet_train)):
        for j in xrange(len(snippet_train[i])):
            dic[snippet_train[i][j]][2] = dic[snippet_train[i][j]][0] / dic[snippet_train[i][j]][1]
            
    for i in xrange(len(snippet_test)):
        score = 0
        for j in xrange(len(snippet_test[i])):
            if snippet_test[i][j] in dic:
                score += dic[snippet_test[i][j]][2]
        myans.append(int(floor(score / len(snippet_test[i]))))
    #print myans
        
    
        
if __name__ == "__main__":
    train_e1 = []
    train_e2 = []
    test_e1 = []
    test_e2 = []
    ans_train = []
    snippet_train = []
    snippet_test = []
    ans = []
    myans = []
    e1_position = []
    e2_position = []
    test_e1, test_e2, train_e1, train_e2, ans_train, snippet_train, snippet_test = parsing()
    assert len(train_e1) == 8000
    assert len(train_e2) == 8000
    assert len(test_e1) == 2717
    assert len(test_e2) == 2717
    assert len(ans_train) == 8000
    assert len(snippet_train) == 8000
    assert len(snippet_test) == 2717


    preprocessing()
    #method1()  # Naive Bayes
    #output()
    #method2()   # best similarity
    #output()
    method3()
    output()
    #method4()
    #output()
#--------------------------------------------------------------------------------------------------------------
#method 1
#Accuracy (calculated for the above confusion matrix) = 639/2717 = 23.52%

#Micro-averaged result (excluding Other):
#P =  595/2550 =  23.33%     R =  595/2263 =  26.29%     F1 =  24.72%

#MACRO-averaged result (excluding Other):
#P =  28.71%	R =  25.54%	F1 =  24.76%
#--------------------------------------------------------------------------------------------------------------
#method 2
#Accuracy (calculated for the above confusion matrix) = 910/2717 = 33.49%
#Micro-averaged result (excluding Other):
#P =  797/2303 =  34.61%     R =  797/2263 =  35.22%     F1 =  34.91%

#MACRO-averaged result (excluding Other):
#P =  33.93%	R =  33.56%	F1 =  32.92%
#--------------------------------------------------------------------------------------------------------------
#method 3
#1017/2717 = 37.43%

#Micro-averaged result (excluding Other):
#P =  970/2541 =  38.17%     R =  970/2263 =  42.86%     F1 =  40.38%

#MACRO-averaged result (excluding Other):
#P =  36.22%	R =  41.50%	F1 =  37.14%
#--------------------------------------------------------------------------------------------------------------
#method 4
#Accuracy (calculated for the above confusion matrix) = 358/2717 = 13.18%

#Micro-averaged result (excluding Other):
#P =  154/1644 =   9.37%     R =  154/2263 =   6.81%     F1 =   7.88%

#MACRO-averaged result (excluding Other):
#P =   1.89%	R =   5.74%	F1 =   2.65%
#--------------------------------------------------------------------------------------------------------------
