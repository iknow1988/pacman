log_file=out.txt
let n=2
echo "Starting training!! [All layouts and "$n" episodes" > $log_file
let episode=1


for((i=1; i<=n; i++)) 
do
    /Users/danielgil/anaconda/envs/Py27/bin/python "/Users/danielgil/Documents/Unimelb/COMP90054 - AI/Assignment/A2/src/comp90054-pacman/pacman-contest/capture.py" -b baselineTeam -r myTeam-DG-RL -l RANDOM$i -n 3 -q >> out.txt
    echo "Episode "$episode" - "RANDOM$i" ... done!" >> $log_file
    let episode=$episode+1
done

for f in ./layouts/* ; do
  /Users/danielgil/anaconda/envs/Py27/bin/python "/Users/danielgil/Documents/Unimelb/COMP90054 - AI/Assignment/A2/src/comp90054-pacman/pacman-contest/capture.py" -r baselineTeam -b myTeam-DG-RL -l $f -n 3 -q >> out.txt
  echo "Episode "$episode" - "$f" ... done!" >> $log_file
  let episode=$episode+1
done