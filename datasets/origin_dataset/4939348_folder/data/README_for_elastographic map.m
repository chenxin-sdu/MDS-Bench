
%============================
%Bmode image display
%(B-mode_inclusion3.mat, B-mode_inclusion7.mat and B-mode_artery.mat)
%please load each of above .mat file in Matlab
image(xyAxis,xyAxis,map)
colormap gray(256)
axis image
set(gca,'FontSize',14,'FontWeight','b','FontName','Times New Roman','LineWidth',1.0)
ylabel('Length (mm)','FontSize',16,'FontName','Times New Roman','FontWeight','b')
xlabel('Length(mm)','FontSize',16,'FontName','Times New Roman','FontWeight','b')

%============================
%wave-amplitude map and wave-velocity map display
%(displacement_inclusion3.mat, displacement_inclusion7.mat and displacement_artery.mat)
%(velocity_inclusion3.mat, velocity_inclusion7.mat and velocity_artery.mat)
%please load each of above .mat file in Matlab

imagesc(xyAxis,xyAxis,map)
colormap jet(256)
axis image
set(gca,'FontSize',14,'FontWeight','b','FontName','Times New Roman','LineWidth',1.0)
ylabel('Length (mm)','FontSize',16,'FontName','Times New Roman','FontWeight','b')
xlabel('Length(mm)','FontSize',16,'FontName','Times New Roman','FontWeight','b')