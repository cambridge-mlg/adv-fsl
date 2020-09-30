import math
import re


def conf_95_to_std(conf_95, n):
    return conf_95 * math.sqrt(n) / 1.96


def std_to_conf_95(std, n):
    return 1.96 * std / math.sqrt(n)


def relative_drop_uncertainty_string(clean, attack, num_tasks):
    # parse the string
    matches_clean = re.findall("\d+\.\d+", clean)
    matches_attack = re.findall("\d+\.\d+", attack)

    return relative_drop_uncertainty(float(matches_clean[0]), float(matches_clean[1]), float(matches_attack[0]), float(matches_attack[1]), num_tasks)


def relative_drop_uncertainty(clean_value, clean_conf_95, attack_value, attack_conf_95, num_tasks):
    """
    This routine computes the 95% confidence interval for the calculation:
    relative_drop_in_accuracy (clean_value - attack_value) / clean_value * 100.0
    :param clean_value:
    :param clean_conf_95:
    :param attack_value:
    :param attack_conf_95:
    :param num_tasks:
    :return:
    """
    # convert 95% confidence intervals to standard deviation
    clean_std = conf_95_to_std(clean_conf_95, num_tasks)
    attack_std = conf_95_to_std(attack_conf_95, num_tasks)

    # do the subtraction
    if clean_value > attack_value:
        diff_value = clean_value - attack_value
    else:
        diff_value = 1e-7
    diff_std = math.sqrt(clean_std * clean_std + attack_std * attack_std)

    # do the division
    out_value = diff_value / clean_value * 100.0
    out_std = out_value * math.sqrt((diff_std / diff_value) ** 2 + (clean_std / clean_value) ** 2)
    out_conf_95 = std_to_conf_95(out_std, num_tasks)

    return out_value, out_conf_95


def main():
    # quick sanity test
    out, out_95 = relative_drop_uncertainty(clean_value=49.0, clean_conf_95=4.12, attack_value=48.3, attack_conf_95=5.5, num_tasks=100)
    print('{:.1f}+/-{:.1f}'.format(out, out_95))

    out, out_95 = relative_drop_uncertainty(clean_value=48.5, clean_conf_95=4.02, attack_value=44.6, attack_conf_95=3.94, num_tasks=100)
    print('{:.1f}+/-{:.1f}'.format(out, out_95))

    out, out_95 = relative_drop_uncertainty(clean_value=73.1, clean_conf_95=1.1, attack_value=9.1, attack_conf_95=0.4, num_tasks=500)
    print('{:.1f}+/-{:.1f}'.format(out, out_95))

    out, out_95 = relative_drop_uncertainty(clean_value=73.5, clean_conf_95=1.0, attack_value=68.4, attack_conf_95=1.1, num_tasks=500)
    print('{:.1f}+/-{:.1f}'.format(out, out_95))

    out, out_95 = relative_drop_uncertainty(clean_value=88.5, clean_conf_95=0.6, attack_value=86.0, attack_conf_95=0.6, num_tasks=500)
    print('{:.1f}+/-{:.1f}'.format(out, out_95))

    out, out_95 = relative_drop_uncertainty(clean_value=57.2, clean_conf_95=0.9, attack_value=54.5, attack_conf_95=0.8, num_tasks=500)
    print('{:.1f}+/-{:.1f}'.format(out, out_95))

    out, out_95 = relative_drop_uncertainty(clean_value=82.2, clean_conf_95=0.5, attack_value=79.6, attack_conf_95=0.6, num_tasks=500)
    print('{:.1f}+/-{:.1f}'.format(out, out_95))

    out, out_95 = relative_drop_uncertainty(clean_value=57.6, clean_conf_95=0.8, attack_value=55.0, attack_conf_95=0.8, num_tasks=500)
    print('{:.1f}+/-{:.1f}'.format(out, out_95))

    out, out_95 = relative_drop_uncertainty(clean_value=81.3, clean_conf_95=0.5, attack_value=79.3, attack_conf_95=0.5, num_tasks=500)
    print('{:.1f}+/-{:.1f}'.format(out, out_95))

    out, out_95 = relative_drop_uncertainty(clean_value=74.4, clean_conf_95=1.0, attack_value=69.7, attack_conf_95=1.0, num_tasks=500)
    print('{:.1f}+/-{:.1f}'.format(out, out_95))

    out, out_95 = relative_drop_uncertainty(clean_value=87.7, clean_conf_95=0.6, attack_value=85.3, attack_conf_95=0.7, num_tasks=500)
    print('{:.1f}+/-{:.1f}'.format(out, out_95))

    out, out_95 = relative_drop_uncertainty_string(clean='88.5+/-0.6', attack='86.0+/-0.6', num_tasks=500)
    print('{:.1f}+/-{:.1f}'.format(out, out_95))


if __name__ == '__main__':
    main()
