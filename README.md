être certain d'avoir la librairie argparse pour traiter les arguments en ligne de commande

pour lancer dans le terminal: 
-----
$ python3 note_detection.py 4chords.wav -g=1 -ca=1
$ python3 note_detection.py Dsharp3.wav -g=1 

pour plus d'infos sur les arguments:
------
$ python3 note_detection.py -h

PART 1:
-------
GOAL: audio of a note or a chord, use fourier transform to recognise notes played
- draw graph
- figure out how to draw amplitude spectrum
- save frequencies based on highest amplitudes with a threshold rule
- distance algorithm to figure out note correspondance

PART 2:
-------
GOAL: chunk analysis
1 - noise reduction gate, high pass filter
2 - catch the start of notes/chords by checking volume increase (derivative of abs(data))
find local minima of gradient(movingavg(left)) correspond to note playing
from this analysis only consider wide enough rectangles and volume past a certain level
3 - sample the audio at the start of notes/chords and before the next one
4 - chord analysis on the sample
5 - result: list of single notes or chords played through time

Liens Utiles:
-------------
https://python-course.eu/applications-python/musical-scores-with-python.php
https://www.google.com/search?q=python+draw+on+sheet+music&oq=python+draw+on+sheet+music+&aqs=chrome..69i57j33i160l5j33i22i29i30j33i15i22i29i30.6604j0j7&sourceid=chrome&ie=UTF-8

"""