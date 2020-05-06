function h= hypothesis(x,theta)
if(theta'*x'>0)
h=1;
else
h=0;
endif;
end;