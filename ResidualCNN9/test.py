import numpy as np

def test(testing_data, input_x, input_p1, input_p2, s, p, dropout_keep_prob, datamanager, sess, num_epoch):
    results = []
    total = 0
    i = 0
    t = 0
    c = 0
    for test in testing_data:
        i += 1
        x_test = datamanager.generate_x(testing_data[test])
        p1, p2 = datamanager.generate_p(testing_data[test])
        y_test = datamanager.generate_y(testing_data[test])
        scores, pre = sess.run([s, p], {input_x: x_test, input_p1:p1, input_p2:p2, dropout_keep_prob: 1.0})
        max_pro = 0
        prediction = -1
        for score in scores:
            score = np.exp(score-np.max(score))
            score = score/score.sum(axis=0)
            score[0] = 0
            pro = score[np.argmax(score)]
            if pro > max_pro and np.argmax(score)!=0:
                max_pro = pro
                prediction = np.argmax(score)
        for i in range(len(testing_data[test])):
            results.append((test, testing_data[test][i].relation.id, max_pro, prediction))
            if testing_data[test][i].relation.id == pre and pre!=0:
                c += 1
            t += 1
            if testing_data[test][i].relation.id != 0:
                total += 1
    print("Correct: "+str(c))
    print("Total: "+str(t))
    print("Accuracy: "+str(float(c)/float(t)))
    results = sorted(results, key=lambda t: t[2])
    results.reverse()
    correct = 0
    f = open("re-9-128_precision_recall_"+str(num_epoch)+".txt", "w")
    print(total)
    for i in range(total):
        if results[i][1] == results[i][3]:
            correct += 1
        if i%100 == 0:
            print("Precision: "+str(float(correct)/float(i+1))+"  Recall: "+str(float(correct)/float(total)))
        f.write(str(float(correct)/float(i+1))+"    "+str(float(correct)/float(total))+"    "+str(results[i][2])+"  "
                +results[i][0]+"  "+str(results[i][3])+"\n")
