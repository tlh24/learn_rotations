!#/usr/bin/bash

gs -dSAFER -dBATCH -dNOPAUSE -dEPSCrop -r100 -sDEVICE=pngalpha -sOutputFile=snr_2000.png snr_2000.db.eps

gs -dSAFER -dBATCH -dNOPAUSE -dEPSCrop -r100 -sDEVICE=pngalpha -sOutputFile=snr_500.png snr_500.db.eps
