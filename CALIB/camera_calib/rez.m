function y = rez(x)
    opt_file = csvread('optimize.txt');
    opt_file_len = length(opt_file);
    RITC = zeros(3,3,opt_file_len);
    RIOB = zeros(3,3,opt_file_len);
    RITO = zeros(3,3,opt_file_len);
    RIBC = zeros(3,3,opt_file_len);
    for i=1:length(opt_file)
        ritc_norm = norm(opt_file(i,1:3));
        RITC(:,:,i) = axang2rotm([opt_file(i,1:3)/ritc_norm,ritc_norm]);
        RIOB(:,:,i) = eul2rotm(opt_file(i,4:6), 'ZYX')';
        RITO(:,:,i) = [cos(x(1)) -sin(x(1)) 0; sin(x(1)) cos(x(1)) 0; 0 0 1];
        RIBC(:,:,i) = [cos(x(2)) -sin(x(2)) 0; sin(x(2)) cos(x(2)) 0; 0 0 1];
        right_hand_side = RITO(:,:,i) * RIOB(:,:,i) * RIBC(:,:,i);
        left_hand_side = RITC(:,:,i);
        R = right_hand_side'*left_hand_side;
        y(i) = 1/2*trace(eye(3) - R);
    end
end