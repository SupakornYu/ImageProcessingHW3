I = imread('Crab.pgm');
imshow(I);
BW1 = edge(I,'Prewitt');
imshowpair(BW1,I,'montage');