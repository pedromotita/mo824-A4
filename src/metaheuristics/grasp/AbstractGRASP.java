/**
 * 
 */
package metaheuristics.grasp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.HashMap;
import java.util.Map;
import java.util.Collections;

import problems.Evaluator;
import solutions.Solution;

/**
 * Abstract class for metaheuristic GRASP (Greedy Randomized Adaptive Search
 * Procedure). It consider a minimization problem.
 * 
 * @author ccavellucci, fusberti
 * @param <E>
 *            Generic type of the element which composes the solution.
 */
public abstract class AbstractGRASP<E> {

	/**
	 * flag that indicates whether the code should print more information on
	 * screen
	 */
	public static boolean verbose = true;

	/**
	 * a random number generator
	 */
	static Random rng = new Random(0);

	/**
	 * the objective function being optimized
	 */
	protected Evaluator<E> ObjFunction;

	/**
	 * the GRASP greediness-randomness parameter
	 */
	protected Double alpha;

	/**
	 * the all possible values for GRASP greediness-randomness parameter used in the reactive constructive method
	 */
	protected Double[] alphaP;

	/**
	 * the best (incumbent) solution cost
	 */
	protected Double bestCost;

	/**
	 * the current solution cost
	 */
	protected Double cost;

	/**
	 * the best solution
	 */
	protected Solution<E> bestSol;

	/**
	 * the current solution
	 */
	protected Solution<E> sol;

	/**
	 * the number of iterations the GRASP main loop executes.
	 */
	protected Integer iterations;

	/**
	 * the Candidate List of elements to enter the solution.
	 */
	protected ArrayList<E> CL;

	/**
	 * the Restricted Candidate List of elements to enter the solution.
	 */
	protected ArrayList<E> RCL;

	/**
	 * Creates the Candidate List, which is an ArrayList of candidate elements
	 * that can enter a solution.
	 * 
	 * @return The Candidate List.
	 */
	public abstract ArrayList<E> makeCL();

	/**
	 * Creates the Restricted Candidate List, which is an ArrayList of the best
	 * candidate elements that can enter a solution. The best candidates are
	 * defined through a quality threshold, delimited by the GRASP
	 * {@link #alpha} greedyness-randomness parameter.
	 * 
	 * @return The Restricted Candidate List.
	 */
	public abstract ArrayList<E> makeRCL();

	/**
	 * Updates the Candidate List according to the current solution
	 * {@link #sol}. In other words, this method is responsible for
	 * updating which elements are still viable to take part into the solution.
	 */
	public abstract void updateCL();

	/**
	 * Creates a new solution which is empty, i.e., does not contain any
	 * element.
	 * 
	 * @return An empty solution.
	 */
	public abstract Solution<E> createEmptySol();

	/**
	 * The GRASP local search phase is responsible for repeatedly applying a
	 * neighborhood operation while the solution is getting improved, i.e.,
	 * until a local optimum is attained.
	 * 
	 * @return An local optimum solution.
	 */
	public abstract Solution<E> localSearch();

	/**
	 * Constructor for the AbstractGRASP class.
	 * 
	 * @param objFunction
	 *            The objective function being minimized.
	 * @param alpha
	 *            The GRASP greediness-randomness parameter (within the range
	 *            [0,1])
	 * @param iterations
	 *            The number of iterations which the GRASP will be executed.
	 */
	public AbstractGRASP(Evaluator<E> objFunction, Double alpha, Integer iterations) {
		this.ObjFunction = objFunction;
		this.alpha = alpha;
		this.iterations = iterations;
	}
	
	/**
	 * The GRASP constructive heuristic, which is responsible for building a
	 * feasible solution by selecting in a greedy-random fashion, candidate
	 * elements to enter the solution.
	 * 
	 * @return A feasible solution to the problem being minimized.
	 */
	public Solution<E> constructiveHeuristic() {

		CL = makeCL();
		RCL = makeRCL();
		sol = createEmptySol();
		cost = Double.POSITIVE_INFINITY;

		/* Main loop, which repeats until the stopping criteria is reached. */
		while (!constructiveStopCriteria()) {

			double maxCost = Double.NEGATIVE_INFINITY, minCost = Double.POSITIVE_INFINITY;
			cost = ObjFunction.evaluate(sol);
			updateCL();

			if (CL.size() > 0) {
				/*
				* Explore all candidate elements to enter the solution, saving the
				* highest and lowest cost variation achieved by the candidates.
				*/
				for (E c : CL) {
					Double deltaCost = ObjFunction.evaluateInsertionCost(c, sol);
					if (deltaCost < minCost)
						minCost = deltaCost;
					if (deltaCost > maxCost)
						maxCost = deltaCost;
				}

				/*
				* Among all candidates, insert into the RCL those with the highest
				* performance using parameter alpha as threshold.
				*/
				for (E c : CL) {
					Double deltaCost = ObjFunction.evaluateInsertionCost(c, sol);
					if (deltaCost <= minCost + alpha * (maxCost - minCost)) {
						RCL.add(c);
					}
				}

				/* Choose a candidate randomly from the RCL */
				int rndIndex = rng.nextInt(RCL.size());
				E inCand = RCL.get(rndIndex);
				CL.remove(inCand);
				sol.add(inCand);
				ObjFunction.evaluate(sol);
				RCL.clear();
			}
		}

		return sol;
	}

	 /** 
	 * Get alpha based on a determined probability.
	 * @param alphaProbability
	 		each element contains the probability of the alpha i being chosen, i = 0, 1, ..., m
			m being the size of array
	 * 
	 * @return Index of a alpha value taking into account the probability
	 */
	public int getAlphaIndex(Double[] alphaProbability) {
		Double sum = 0.0;
		ArrayList<Double> cumulativeSum = new ArrayList<>();
		for (Double p : alphaProbability) {
			sum += p;
			cumulativeSum.add(sum);
		}
		int alphaIndex = Arrays.binarySearch(cumulativeSum.toArray(), rng.nextDouble() * sum);

		return Math.abs(alphaIndex);
	}

	/**
	 * The GRASP constructive reactive heuristic, which is responsible for building a
	 * feasible solution by selecting in a greedy-random fashion, candidate
	 * elements to enter the solution and changing the value of alpha in a reactive way based
	 * on previous iterations.
	 * 
	 * @return A feasible solution to the problem being minimized.
	 */
	public Solution<E> constructiveReactiveHeuristic() {

		CL = makeCL();
		RCL = makeRCL();
		sol = createEmptySol();
		cost = Double.POSITIVE_INFINITY;


		/* Set up variable that will be used to determine the best values for alpha*/
		int numPossibleAlpha = 10;
		Double[] alphas = new Double[numPossibleAlpha];
		Double[] sumCostUsingAlpha = new Double[numPossibleAlpha];
		Integer[] averageCostUsingAlphaCount = new Integer[numPossibleAlpha];
		Double[] alphaProbability = new Double[numPossibleAlpha];
		double previousAlpha = 0;
		for (int i=0; i < numPossibleAlpha; i++) {
			// Initially probability of any alpha is pi = 1/m , for i = 0,1 .., m 
			// m being the size of the set of all possible alpha values
			alphaProbability[i] = 1/ (double) numPossibleAlpha;

			//Set all possible alpha values
			previousAlpha += 1/ (double) numPossibleAlpha;
			alphas[i] = previousAlpha;

			// Set the average function cost for a given i alpha 
			sumCostUsingAlpha[i] = cost;
			averageCostUsingAlphaCount[i] = 0;
		}

		/* Main loop, which repeats until the stopping criteria is reached. */
		while (!constructiveStopCriteria()) {

			double maxCost = Double.NEGATIVE_INFINITY, minCost = Double.POSITIVE_INFINITY;
			cost = ObjFunction.evaluate(sol);
			updateCL();
			int alphaIndex = Math.min(getAlphaIndex(alphaProbability), numPossibleAlpha-1);
			alpha = alphas[alphaIndex];

			/*
			 * Explore all candidate elements to enter the solution, saving the
			 * highest and lowest cost variation achieved by the candidates.
			 */
			for (E c : CL) {
				Double deltaCost = ObjFunction.evaluateInsertionCost(c, sol);
				if (deltaCost < minCost)
					minCost = deltaCost;
				if (deltaCost > maxCost)
					maxCost = deltaCost;
			}

			/*
			 * Among all candidates, insert into the RCL those with the highest
			 * performance using parameter alpha as threshold.
			 */
			for (E c : CL) {
				Double deltaCost = ObjFunction.evaluateInsertionCost(c, sol);
				if (deltaCost <= minCost + alpha * (maxCost - minCost)) {
					RCL.add(c);
				}
			}

			if (!RCL.isEmpty()){
				/* Choose a candidate randomly from the RCL */
				int rndIndex = rng.nextInt(RCL.size());
				E inCand = RCL.get(rndIndex);
				CL.remove(inCand);
				sol.add(inCand);
				ObjFunction.evaluate(sol);
				RCL.clear();
			}
			
			// Based on the new values for cost updateAlphaProbabilities
			averageCostUsingAlphaCount[alphaIndex] += 1;
			sumCostUsingAlpha[alphaIndex] += cost;

			double averageCostUsingAlpha = sumCostUsingAlpha[alphaIndex] / averageCostUsingAlphaCount[alphaIndex];

			double auxSum = 0;
			for (int i=0; i < numPossibleAlpha; i++) {
				auxSum += sumCostUsingAlpha[i] / averageCostUsingAlphaCount[i];
			}

			alphaProbability[alphaIndex] = averageCostUsingAlpha / auxSum;

		}

		return sol;
	}

	public Solution<E> constructiveSampleGreedyHeuristic(Double p) {
		CL = makeCL();
		RCL = makeRCL();
		sol = createEmptySol();
		cost = Double.POSITIVE_INFINITY;
		int numberOfCandidatesToSelect = Math.min(CL.size(), p.intValue());
		
		while (!constructiveStopCriteria()) {

			double minCost = Double.POSITIVE_INFINITY;
			cost = ObjFunction.evaluate(sol);
			updateCL();

			if (CL.size() > 0) {
				/*
				* Among all candidates, randomly sample min{p, |C|} elements and insert into the RCL those
				* performance using parameter alpha as threshold.
				*/
				ArrayList<E> selectedCandidates = new ArrayList<E>();
				for (int i = 0; i < numberOfCandidatesToSelect; i++) {
					// generating the index using Math.random()
					Random random = new Random();
					int randomIndex = random.nextInt(CL.size());
					// int index = (int)(Math.random() * CL.size());
					selectedCandidates.add(CL.get(randomIndex));
				}

				Map<Double, E> candidatesAndCost = new HashMap<Double, E>();
				for (E c : selectedCandidates) {
					Double cost = ObjFunction.evaluateInsertionCost(c, sol);
					candidatesAndCost.put(cost, c);
				}
				
				minCost = Collections.min(candidatesAndCost.keySet());
				RCL.add(candidatesAndCost.get(minCost));

				if (!RCL.isEmpty()){
					/* Choose a candidate randomly from the RCL */
					int rndIndex = rng.nextInt(RCL.size());
					E inCand = RCL.get(rndIndex);
					CL.remove(inCand);
					// numberOfCandidatesToSelect = (int)Math.min(p, CL.size());
					sol.add(inCand);
					ObjFunction.evaluate(sol);
					RCL.clear();
				}
			}
		}

		return sol;
	}

	/**w
	 * The GRASP mainframe. It consists of a loop, in which each iteration goes
	 * through the constructive heuristic and local search. The best solution is
	 * returned as result.
	 * 
	 * @return The best feasible solution obtained throughout all iterations.
	 */
	public Solution<E> solve() {

		bestSol = createEmptySol();
		for (int i = 0; i < iterations; i++) {
			constructiveHeuristic();
			// constructiveReactiveHeuristic();
			// constructiveSampleGreedyHeuristic((double)10);
			localSearch();
			if (bestSol.cost > sol.cost) {
				bestSol = new Solution<E>(sol);
				if (verbose)
					System.out.println("(Iter. " + i + ") BestSol = " + bestSol);
			}
		}

		return bestSol;
	}

	/**
	 * A standard stopping criteria for the constructive heuristic is to repeat
	 * until the current solution improves by inserting a new candidate
	 * element.
	 * 
	 * @return true if the criteria is met.
	 */
	public Boolean constructiveStopCriteria() {
		return (cost > sol.cost) ? false : true;
	}

}
