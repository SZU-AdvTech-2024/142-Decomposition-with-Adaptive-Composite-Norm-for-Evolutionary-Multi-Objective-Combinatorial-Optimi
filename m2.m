classdef m2 < ALGORITHM
    % <multi/many> <binary/permutation>
   
    methods
        function main(Algorithm, Problem)
            %% Parameter Initialization
            [Tm, Tr, nr, RC, drho_m, L1] = Algorithm.ParameterSet(0.1, 0.1, 0.1, 0, 0.1, 100);
            
            %% Initialize
            [W, Problem.N] = UniformPoint(Problem.N, Problem.M);
            
            Tm = ceil(Problem.N * Tm);
            Tr = ceil(Problem.N * Tr);
            nr = ceil(Problem.N * nr);

            % Find the neighbours based on weight vectors
            B = pdist2(W, W);
            [~, B] = sort(B, 2);
            Bm = B(:, 1:Tm);
            Br = B(:, 1:Tr);

            % Generate initial population
            Population = Problem.Initialization();
            FontNo = NDSort(Population.objs, 1);
            Archive = Population(FontNo == 1);
            [~, ia] = unique(Archive.objs, 'row');
            Archive = Archive(ia);

            zmin = min(Population.objs, [], 1);
            zmax = max(Population.objs, [], 1);
            z = zmin - RC * (zmax - zmin);

            % Initialize adaptive parameters
            if drho_m <= -1 && drho_m >= -2
                rho = (2 + drho_m) * ones(Problem.N, 1);
                drho_l = 0;
                drho_n = 0;
            elseif drho_m > -1
                rho = ones(Problem.N, 1);
                drho_l = drho_m;
                drho_n = drho_m;
            end

            L = L1;
            Eta = zeros(Problem.N, L);
            eta = zeros(Problem.N, 1);
            pointer_Eta = 0;
            h = true(Problem.N, 1);

            subp_test = randperm(Problem.N, 1);
            pointer_Zeta = 0;
            Zeta = zeros(Problem.N, 3 * length(subp_test));

            %% Optimization Loop
            while Algorithm.NotTerminated(Archive)
                pointer_Eta = pointer_Eta + 1;
                Eta(h, pointer_Eta) = eta(h);
                eta(h) = 0;
                index = ~h;
                if pointer_Eta == 1
                    Eta(index, pointer_Eta) = Eta(index, L1);
                else
                    Eta(index, pointer_Eta) = Eta(index, pointer_Eta - 1);
                end
                eta(index) = eta(index) + 1;
                index = Eta(:, pointer_Eta) < eta;
                Eta(index, pointer_Eta) = eta(index);
                h = false(Problem.N, 1);
                if pointer_Eta >= L1
                    pointer_Eta = 0;
                end

                subp_test = randi(Problem.N);

                O_test = repmat(SOLUTION(), 1, 3 * Tm * length(subp_test));
                rho_last = min(max(rho + drho_l, 0), 1);
                rho_next = min(max(rho - drho_n, 0), 1);
                Pop_tmp = [Population Archive];
                objs = Pop_tmp.objs;
                Pop_test_last = repmat(SOLUTION(), 1, Problem.N);
                Pop_test_cur = repmat(SOLUTION(), 1, Problem.N);
                Pop_test_next = repmat(SOLUTION(), 1, Problem.N);
                count = 0;
                for j = subp_test
                    for jj = Bm(j, :)
                        g = CompositeNorm(objs, z, W(jj, :), rho_last(jj));
                        [~, I] = min(g);
                        Pop_test_last(jj) = Pop_tmp(I);
                        g = CompositeNorm(objs, z, W(jj, :), rho(jj));
                        [~, I] = min(g);
                        Pop_test_cur(jj) = Pop_tmp(I);
                        g = CompositeNorm(objs, z, W(jj, :), rho_next(jj));
                        [~, I] = min(g);
                        Pop_test_next(jj) = Pop_tmp(I);
                    end

                    for jj = 1: Tm
                        P = [j, Bm(j, jj)];
                        Offspring_cur = OperatorGAhalf(Problem, Pop_test_cur(P(1:2)));
                        Offspring_last = OperatorGAhalf(Problem, Pop_test_last(P(1:2)));
                        Offspring_next = OperatorGAhalf(Problem, Pop_test_next(P(1:2)));
                        O_test((count * Tm) + jj + [0, length(O_test) / 3, length(O_test) / 3 * 2]) = ...
                            [Offspring_last, Offspring_cur, Offspring_next];
                    end
                    count = count + 1;
                end

                zmax = max(Population.objs, [], 1);
                zmin = min([zmin; O_test.objs]);
                z = zmin - RC * (zmax - zmin);

                count = 0;
                for j = 1: length(subp_test)
                    R = Br(subp_test(j), randperm(Tr));
                    for jj = 1: Tm
                        index_last = count * Tm + jj;
                        index_cur = count * Tm + length(O_test) / 3 + jj;
                        index_next = count * Tm + length(O_test) / 3 * 2 + jj;

                        g_old_last = CompositeNorm(Pop_test_last(R).objs, z, W(R, :), rho_last(R));
                        g_old_cur = CompositeNorm(Pop_test_cur(R).objs, z, W(R, :), rho(R));
                        g_old_next = CompositeNorm(Pop_test_next(R).objs, z, W(R, :), rho_next(R));
                        g_new_last = CompositeNorm(O_test(index_last).obj, z, W(R, :), rho_last(R));
                        g_new_cur = CompositeNorm(O_test(index_cur).obj, z, W(R, :), rho(R));
                        g_new_next = CompositeNorm(O_test(index_next).obj, z, W(R, :), rho_next(R));

                        index = g_old_last > g_new_last;
                        Zeta(R(index), 1) = Zeta(R(index), 1) + 1;
                        index = g_old_cur > g_new_cur;
                        Zeta(R(index), 2) = Zeta(R(index), 2) + 1;
                        index = g_old_next > g_new_next;
                        Zeta(R(index), 3) = Zeta(R(index), 3) + 1;

                        g_old = CompositeNorm(Population(R).objs, z, W(R, :), rho(R));
                        g_new = CompositeNorm(O_test(index_last).obj, z, W(R, :), rho(R));
                        I_R = R(find(g_old >= g_new, nr));
                        Population(I_R) = O_test(index_last);
                        I_imp = intersect(R(g_old > g_new), I_R);
                        h(I_imp) = true;

                        g_old = CompositeNorm(Population(R).objs, z, W(R, :), rho(R));
                        g_new = CompositeNorm(O_test(index_cur).obj, z, W(R, :), rho(R));
                        I_R = R(find(g_old >= g_new, nr));
                        Population(I_R) = O_test(index_cur);
                        I_imp = intersect(R(g_old > g_new), I_R);
                        h(I_imp) = true;

                        g_old = CompositeNorm(Population(R).objs, z, W(R, :), rho(R));
                        g_new = CompositeNorm(O_test(index_next).obj, z, W(R, :), rho(R));
                        I_R = R(find(g_old >= g_new, nr));
                        Population(I_R) = O_test(index_next);
                        I_imp = intersect(R(g_old > g_new), I_R);
                        h(I_imp) = true;
                    end
                    count = count + 1;
                end

                pointer_Zeta = pointer_Zeta + 1;
                if pointer_Zeta >= L
                    pointer_Zeta = 0;

                    Pop_tmp = [Population Archive];
                    objs = Pop_tmp.objs;
                    for j = 1: Problem.N
                        tmp = zeros(1, 3);
                        tmp(3) = sum(Zeta(B(j, 1: ceil(Problem.N / 2)), 1));
                        tmp(2) = sum(Zeta(B(j, 1: ceil(Problem.N / 2)), 2));
                        tmp(1) = sum(Zeta(B(j, 1: ceil(Problem.N / 2)), 3));
                        [~, I] = max(tmp);
                        switch I
                            case 3
                                rho(j) = rho_last(j);
                            case 1
                                rho(j) = rho_next(j);
                        end

                        if I ~= 2
                            g = CompositeNorm(objs, z, W(j, :), rho(j));
                            [~, I] = min(g);
                            Population(j) = Pop_tmp(I);
                        end
                    end

                    Zeta(:) = 0;
                end

                O = repmat(SOLUTION(), 1, Problem.N);
                for j = randperm(Problem.N)
                    P = [j, Bm(j, randperm(Tm, 1))];
                    R = Br(j, randperm(Tr));

                    Offspring = OperatorGAhalf(Problem, Population(P(1:2)));
                    O(j) = Offspring;

                    zmin = min(zmin, Offspring.obj);
                    z = zmin - RC * (zmax - zmin);

                    g_old = CompositeNorm(Population(R).objs, z, W(R, :), rho(R));
                    g_new = CompositeNorm(Offspring.obj, z, W(R, :), rho(R));
                    I_R = R(find(g_old >= g_new, nr));
                    Population(I_R) = Offspring;
                    I_imp = intersect(R(g_old > g_new), I_R);
                    h(I_imp) = true;
                end

                Archive = UpdateArchive(Archive, [O_test, O], Problem);
            end
        end
    end
end

function g = CompositeNorm(objs, z, W, rho)
    g = rho .* sum((objs - z) .* W, 2) + (1 - rho) .* max((objs - z) .* W, [], 2);
end

function Archive = UpdateArchive(Archive, Z, Problem)
    Archive = [Archive, Z];
    [~, ia] = unique(Archive.objs, 'row');
    Archive = Archive(ia);
    FrontNo = NDSort(Archive.objs, 1);
    Next = FrontNo == 1;
    Archive = Archive(Next);
    N_archive = length(Archive);
    if N_archive > Problem.N
        Next = true(1, N_archive);
        Del = Truncation(Archive.objs, N_archive - Problem.N);
        Next(Del) = false;
        Archive = Archive(Next);
    end
end

function Del = Truncation(PopObj, K)
    Distance = pdist2(PopObj, PopObj);
    Distance(logical(eye(length(Distance)))) = inf;
    Del = false(1, size(PopObj, 1));
    while sum(Del) < K
        Remain = find(~Del);
        Temp = sort(Distance(Remain, Remain), 2);
        [~, Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = true;
    end
end
