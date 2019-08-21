import os, random
import shutil

path = "../Dataset/Train/"
subdirect = os.listdir(path)
file = []
cou = 0
fold = 4 
pathdst = '../Test/testdir/'
for i in range(fold):
    os.mkdir(pathdst + 'fold' + str(i+1))
    os.mkdir(pathdst + 'fold' + str(i+1) + '/Train')
    os.mkdir(pathdst + 'fold' + str(i+1) + '/Validation')
    os.mkdir(pathdst + 'fold' + str(i+1) + '/Test')

for eachdirect in subdirect:
    fullpath = os.path.join(path, eachdirect)
    file.clear()

    for f in os.listdir(fullpath):
        file.append(os.path.join(fullpath, f))

    random.shuffle(file)
    eachset = len(file)/fold
    for i in range(fold):
        os.mkdir(pathdst + 'fold' + str(i+1) + '/Train/' + str(cou+1))
        os.mkdir(pathdst + 'fold' + str(i+1) + '/Validation/' + str(cou+1))
        os.mkdir(pathdst + 'fold' + str(i+1) + '/Test/' + str(cou+1))

    for i in range(len(file)): # train file lost 1 file - (128)
        if (i <= eachset):
            shutil.copy2(file[i], pathdst + 'fold1/train/' + str(cou+1))
            shutil.copy2(file[i+int(eachset)], pathdst + 'fold1/train/' + str(cou+1))

            #validation
            shutil.copy2(file[i], pathdst + 'fold3/validation/' + str(cou+1))

            #test
            shutil.copy2(file[i], pathdst + 'fold2/test/' + str(cou+1))
        elif ((i > eachset and i <= eachset*2)):
            shutil.copy2(file[i], pathdst + 'fold2/train/' + str(cou+1))
            shutil.copy2(file[i+int(eachset)], pathdst + 'fold2/train/' + str(cou+1))

            #validation
            shutil.copy2(file[i], pathdst + 'fold4/validation/' + str(cou+1))

            #test
            shutil.copy2(file[i], pathdst + 'fold3/test/' + str(cou+1))
        elif ((i > eachset*2 and i <= eachset*3)):
            shutil.copy2(file[i], pathdst + 'fold3/train/' + str(cou+1))
            if ( i + int(eachset) < len(file)):
                shutil.copy2(file[i+int(eachset)], pathdst + 'fold3/train/' + str(cou+1))
            
            #validation
            shutil.copy2(file[i], pathdst + 'fold1/validation/' + str(cou+1))

            #test
            shutil.copy2(file[i], pathdst + 'fold4/test/' + str(cou+1))
        elif ((i > eachset*3)):
            shutil.copy2(file[i], pathdst + 'fold4/train/' + str(cou+1))
            shutil.copy2(file[i%int(eachset)], pathdst + 'fold4/train/' + str(cou+1))

            #validation
            shutil.copy2(file[i], pathdst + 'fold2/validation/' + str(cou+1))

            #test
            shutil.copy2(file[i], pathdst + 'fold1/test/' + str(cou+1))
    # break

    cou += 1