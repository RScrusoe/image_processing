load('out1.mat')
size(vect)

figure(1);
clf;
imshow(vect, []);
axis image;
colormap('jet');
colorbar;

imwrite(vect,'out1.png');

title(strcat("Depth Map"));
