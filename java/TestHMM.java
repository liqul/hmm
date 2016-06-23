
public class TestHMM {
    static int[] states = new int[] { 0, 1, 2, 3};
    static int[] observations = new int[] {1,0,0,2};
    static double[][] dumy_dist(int[] obs){
        double[][] vecs = new double[obs.length][states.length];
        for (int i = 0; i < obs.length; i++){
            vecs[i][0] = 0;
            vecs[i][1] = 0;
            vecs[i][2] = 0;
            vecs[i][3] = 0;
            vecs[i][obs[i]] = 1;
        }
        return vecs;
    }
    static double[] start_probability = new double[] {0.25, 0.25, 0.25, 0.25};
    static double[][] transititon_probability = new double[][] {
        {0.89,0.06,0.03,0.03}, {0.05,0.84,0.05,0.05}, {0.01,0.07,0.80,0.03}, {0.11,0.11,0.11,0.67}
    };
    static double[][] emission_probability = new double[][] {
        {0.8, 0.01, 0.02, 0.17}, {0.03, 0.9, 0.03, 0.03}, {0.01, 0.07, 0.8, 0.03}, {0.4, 0.01, 0.02, 0.57}
    };
    public static void main(String args[]){
        int[] result = Viterbi.compute(observations, states, start_probability, transititon_probability, emission_probability);
        for (int i = 0; i < result.length; i++){
            System.out.print(result[i] + " ");
        }
        System.out.println();
        double[][] obs_probs = dumy_dist(observations);
        int[] result2 = Viterbi.compute2(obs_probs, states, start_probability, transititon_probability, emission_probability);
        for (int i = 0; i < result2.length; i++){
            System.out.print(result2[i] + " ");
        }
        System.out.println();
    }
}

class Viterbi {
  /**
   * 求解HMM模型
   * @param obs 观测序列
   * @param states 隐状态
   * @param start_p 初始概率（隐状态）
   * @param trans_p 转移概率（隐状态）
   * @param emit_p 发射概率 （隐状态表现为显状态的概率）
   * @return 最可能的序列
   */
  public static int[] compute(int[] obs, int[] states, double[] start_p, double[][] trans_p, double[][] emit_p) {
    double[][] V = new double[obs.length][states.length];
    int[][] path = new int[states.length][obs.length];

    for (int y : states) {
      V[0][y] = start_p[y] * emit_p[y][obs[0]];
      path[y][0] = y;
    }

    for (int t = 1; t < obs.length; ++t) {
      int[][] newpath = new int[states.length][obs.length];

      for (int y : states) {
        double prob = -1;
        int state;
        for (int y0 : states) {
          double nprob = V[t - 1][y0] * trans_p[y0][y] * emit_p[y][obs[t]];
          if (nprob > prob) {
            prob = nprob;
            state = y0;
            // 记录最大概率
            V[t][y] = prob;
            // 记录路径
            System.arraycopy(path[state], 0, newpath[y], 0, t);
            newpath[y][t] = y;
          }
        }
      }

      path = newpath;
    }

    double prob = -1;
    int state = 0;
    for (int y : states) {
      if (V[obs.length - 1][y] > prob) {
        prob = V[obs.length - 1][y];
        state = y;
      }
    }

    return path[state];
  }

  public static double inner_product(double[] v1, double[] v2) {
    double sum = 0;
    for (int i = 0; i < v1.length; i++){
        sum += v1[i] * v2[i];
    }
    return sum;
  }

  /**
   * 求解HMM模型
   * @param obs 观测序列
   * @param states 隐状态
   * @param start_p 初始概率（隐状态）
   * @param trans_p 转移概率（隐状态）
   * @param emit_p 发射概率 （隐状态表现为显状态的概率）
   * @return 最可能的序列
   */
  public static int[] compute2(double[][] obs_probs, int[] states, double[] start_p, double[][] trans_p, double[][] emit_p) {
    double[][] V = new double[obs_probs.length][states.length];
    int[][] path = new int[states.length][obs_probs.length];

    for (int y : states) {
      V[0][y] = start_p[y] * inner_product(emit_p[y], obs_probs[0]);
      path[y][0] = y;
    }

    for (int t = 1; t < obs_probs.length; ++t) {
      int[][] newpath = new int[states.length][obs_probs.length];

      for (int y : states) {
        double prob = -1;
        int state;
        for (int y0 : states) {
          double nprob = V[t - 1][y0] * trans_p[y0][y] * inner_product(emit_p[y], obs_probs[t]);
          if (nprob > prob) {
            prob = nprob;
            state = y0;
            // 记录最大概率
            V[t][y] = prob;
            // 记录路径
            System.arraycopy(path[state], 0, newpath[y], 0, t);
            newpath[y][t] = y;
          }
        }
      }

      path = newpath;
    }

    double prob = -1;
    int state = 0;
    for (int y : states) {
      if (V[obs_probs.length - 1][y] > prob) {
        prob = V[obs_probs.length - 1][y];
        state = y;
      }
    }

    return path[state];
  }
}


