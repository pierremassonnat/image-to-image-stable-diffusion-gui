cd resultats; ffmpeg -f image2 -i %d.png -r 30 -vcodec mpeg4 -b 15000k ../compilation.mp4;
