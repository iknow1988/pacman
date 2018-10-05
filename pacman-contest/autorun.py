import os
from time import sleep
from os import listdir

i = 0
n = 1
folder = "C:\Users\Kazi Abir Adnan\Google Drive (kadnan@student.unimelb.edu.au)\Unimelb Courses\Semester 2\COMP90054\Assignments\Project 2\pacman-contest\layouts2"

l=os.listdir(folder)
li=[x.split('.')[0] for x in l]

for layout in li:
    print "**********************************************************"
    print "\n", layout, " . We will play for ", str(n), " times"
    sleep(1)
    fileName = r'"C:\Users\Kazi Abir Adnan\Google Drive (kadnan@student.unimelb.edu.au)\Unimelb Courses\Semester 2\COMP90054\Assignments\Project 2\pacman-contest\capture.py"'
    args = " -b myTeam-Kazi -r baselineTeam -l " + layout + " -z 0.5"
    # args = args +" -Q -n "+ str(n)
    os.system("python " + fileName + args)
    print "LEFT : ", str(len(li) - (i + 1))
    print "**********************************************************"
    i = i + 1