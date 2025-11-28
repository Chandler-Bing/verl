def rank_reward(solution_strs, mob3d30s, n):
    solution_strs = [float(solution_str) / 100 for solution_str in solution_strs]
    # solution_strs = [float(solution_str) / 100 for solution_str in solution_strs]
    mob3d30s = [int(mob3d30) for mob3d30 in mob3d30s]

    print(f'{len(solution_strs)=}, {len(mob3d30s)=},{mob3d30s.count(1)=},{mob3d30s.count(0)=}')

    rewards = [0] * len(solution_strs)
    counts = [0] * len(solution_strs)
    group_avg = []
    for i in range(0, len(solution_strs), n):
        group_avg.append(sum(solution_strs[i:i + n]) / n)
    for i in range(len(solution_strs)):
        sample_mob3d30 = mob3d30s[i]
        sample_solution = solution_strs[i]
        group_index = i // n
        for j in range(len(group_avg)):
            group_mob3d30 = mob3d30s[j * n]
            if j == group_index or sample_mob3d30 == group_mob3d30:
                continue
            if sample_mob3d30 > group_mob3d30:
                if sample_solution > group_avg[j]:
                    rewards[i] += 1
            elif sample_mob3d30 < group_mob3d30:
                if sample_solution < group_avg[j]:
                    rewards[i] += 1
            else:
                pass
            counts[i] += 1

    rewards = [rewards[i] / counts[i] if counts[i] > 0 else 0 for i in range(len(rewards))]
    return rewards


if __name__ == "__main__":
    solution_strs = ['90', '20', '86', '60', '50', '40']
    mob3d30s = ['1', '1', '0', '0', '1', '1']
    n = 2
    rewards = rank_reward(solution_strs, mob3d30s, n)
    print(rewards)