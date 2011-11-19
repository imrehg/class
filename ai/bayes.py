"""
Naive Bayes with Laplacean smoothing calculation, as in the AI-class
"""
from __future__ import division

def vocab(cl):
    """ Generate vocabularies, total an class-separated as well """
    v = {}
    clv = {}
    for c in cl:
        vc = {}
        items = cl[c]
        for item in items:
            words = item.split()
            for w in words:
                if w in v:
                    v[w] += 1
                else:
                    v[w] = 1
                if w in vc:
                    vc[w] += 1
                else:
                    vc[w] = 1
        clv[c] = vc
    return v, clv

def clprior(cl, tclass, k=1):
    """ Class prior """
    target, total = 0, 0
    nclass = len(cl.keys())
    for c in cl:
        n = len(cl[c])
        total += n
        if c == tclass:
            target = n
    prob = (target + k) / (total + k * nclass)
    return prob

def itemprior(cl, tclass, titem, k=1):
    """ conditional item """
    target, total = 0, 0
    tv, cv = vocab(cl)
    nclass = len(tv.keys())
    for item in cv[tclass].keys():
        total += cv[tclass][item]
        if item == titem:
            target = cv[tclass][item]
    prob = (target + k) / (total + k * nclass)
    return prob

def classpost(cl, tclass, titems, k=1):
    """ Class posterior conditional on items"""
    target, total = 0, 0
    titeml = titems.split()
    for i in cl.keys():
        p1 = 1
        for ti in titeml:
            p1 *= itemprior(cl, i, ti, k)
        p2 = clprior(cl, i, k)
        total += p1 * p2
        if i == tclass:
            target = p1 * p2
    prob = target / total
    return prob

if __name__ == "__main__":
    print "Naive Bayes with Laplacean-smoothing"

    ###### From the homework
    print "Homework:"
    cl= {'movie': ["a perfect world", "my perfect woman", "pretty woman"],
         'song': ["a perfect day", 'electric storm', 'another rainy day'],
         }
    k = 1
    tv, cv = vocab(cl)
    ctest = ["movie", "song"]
    for c in ctest:
        print "P(\"%s\") = %.4f" %(c, clprior(cl, c, k))

    contest = [("perfect", "movie"), ("perfect", "song"),
               ("storm", "movie"), ("storm", "song")]
    for c in contest:
        print "P(\"%s\"|\"%s\") = %.4f" %(c[0], c[1], itemprior(cl, c[1], c[0], k))
    
    classtest = [("movie", "perfect storm")]
    for c in classtest:
        print "P(\"%s\"|\"%s\") = %.4f" %(c[0], c[1], classpost(cl, c[0], c[1], k))

    print "\n", "="*20, "\n"

    ###### From the exam
    print "Exam:"
    cl= {'old': ["top gun", "shy people", "top hat"],
         'new': ["top gear", "gun shy"],
         }
    k = 1
    tv, cv = vocab(cl)
    print "Vocabulary size: %d" %(len(tv.keys()))

    ctest = ["old"]
    for c in ctest:
        print "P(\"%s\") = %.4f" %(c, clprior(cl, c, k))

    contest = [("top", "old")]
    for c in contest:
        print "P(\"%s\"|\"%s\") = %.4f" %(c[0], c[1], itemprior(cl, c[1], c[0], k))
    
    classtest = [("old", "top")]
    for c in classtest:
        print "P(\"%s\"|\"%s\") = %.4f" %(c[0], c[1], classpost(cl, c[0], c[1], k))
