import os
from time import sleep
from os import listdir

i = 0
n = 2
folder = "C:\Users\Kazi Abir Adnan\Google Drive (kadnan@student.unimelb.edu.au)\Unimelb Courses\Semester 2\COMP90054\Assignments\Project 2\pacman-contest\layouts"

l=os.listdir(folder)
li=[x.split('.')[0] for x in l]

for layout in li:
    sleep(2)
    fileName = r'"C:\Users\Kazi Abir Adnan\Google Drive (kadnan@student.unimelb.edu.au)\Unimelb Courses\Semester 2\COMP90054\Assignments\Project 2\pacman-contest\capture.py"'
    args = " -r myTeam-Kazi -b baselineTeam -l " + layout +" -n 10 -Q"
    os.system("python " + fileName + args)