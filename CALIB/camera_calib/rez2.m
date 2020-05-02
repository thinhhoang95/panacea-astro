function y = rez2(x)
    opt_file = csvread('optimize.txt');
    opt_file_len = length(opt_file);
    RITPT = [1 0 0; 0 -1 0; 0 0 -1];
    RITPTZ = [-1,0,0;0,-1,0;0,0,1];
    RITC = zeros(3,3,opt_file_len); % Ri2->C
    RIOB = zeros(3,3,opt_file_len); % Ri1->B
    RITO = zeros(3,3,opt_file_len); % Ri2->i1
    RIBC = zeros(3,3,opt_file_len); % RB->C

    for i=1:length(opt_file)
        ritc_norm = norm(opt_file(i,1:3));
        RITC(:,:,i) = axang2rotm([opt_file(i,1:3)/ritc_norm,ritc_norm]);
        RIOB(:,:,i) = eul2rotm(opt_file(i,4:6), 'ZYX')';
        RITO(:,:,i) = [cos(x(1)) -sin(x(1)) 0; sin(x(1)) cos(x(1)) 0; 0 0 1];
        % RITO(:,:,i) = axang2rotm([x(1:3)/norm(x(1:3)), norm(x(1:3))]);
        % RIBC(:,:,i) = [cos(x(4)) -sin(x(4)) 0; sin(x(4)) cos(x(4)) 0; 0 0 1];
        RIBC(:,:,i) = [cos(x(2)) -sin(x(2)) 0; sin(x(2)) cos(x(2)) 0; 0 0 1];
        % RIBC(:,:,i) = axang2rotm([x(4:6)/norm(x(4:6)), norm(x(4:6))]);
        right_hand_side = RITPT * RITO(:,:,i) * RIOB(:,:,i)' * RIBC(:,:,i);
        left_hand_side = RITC(:,:,i);
        %fprintf('Image ')
        %disp(i)
        %fprintf('R is:')
        %disp(RIOB(:,:,i));
        %fprintf('rvec is: ')
        %disp(left_hand_side);
        R = left_hand_side'*right_hand_side;
        R_a = rotm2axang(R);
        axang = R_a(4) * R_a(1:3);
        y(i) = R_a(4);
        % y(i) = 1/2*trace(eye(3) - R);
    end
end