#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class Probabilities:
    """
    Class to manage observational and experimental probability distributions
    and provide helper methods for all probability calculations.
    """

    def __init__(self, p_obs, p_exp):
        """
        Args:
            p_obs (dict): dictionary of observational probabilities. 
            The keys are tuples in the form ('y', 'x'), where 'y' is an outcome 
            and 'x' is a treatmet. Example: {('y1', 'x1'): 0.5 }
            
            p_exp (dict): dictionary of experimental probabilities. 
            The keys are tuples in the form ('x', 'y'), where 'y' is an outcome 
            and 'x' is a treatmet. Example: {('x1', 'y1'): 0.5 }

        Returns:
            None.

        """
        self.p_obs = p_obs  # P(x, y) observational data
        self.p_exp = p_exp  # P(y|do(x)) experimental data
        self._cache = {}

        # Get all unique x and y values from the data
        self.x_values = sorted(
            list(set(k[0] for k in p_obs.keys()) | set(k[1] for k in p_exp.keys())))
        self.y_values = sorted(
            list(set(k[1] for k in p_obs.keys()) | set(k[0] for k in p_exp.keys())))

        self.verifiy_probability_distributions()
        self.check_data_completeness()
        
    def get_p_obs_x(self, x_val):
        """
        Calculate marginal probability P(x) from observational data

        Args:
            x_val (str): treatment.

        Returns:
            float: marginal probability.

        """
        if not (x_val in self.x_values):
            raise ValueError(f"Error: unknown treatment {x_val}")

        if ('p_x', x_val) in self._cache:
            return self._cache[('p_x', x_val)]
        prob = sum(self.p_obs.get((x_val, y), 0) for y in self.y_values)
        self._cache[('p_x', x_val)] = prob
        return prob

    def get_p_obs_y(self, y_val):
        """
        Calculate marginal probability P(y) from observational data.

        Args:
            y_val (str): outcome.

        Returns:
            float: marginal probability.

        """
        if not (y_val in self.y_values):
            raise ValueError(f"Error: unknown outcome {y_val}")

        if ('p_y', y_val) in self._cache:
            return self._cache[('p_y', y_val)]
        prob = sum(self.p_obs.get((x, y_val), 0) for x in self.x_values)
        self._cache[('p_y', y_val)] = prob
        return prob
    
    def verifiy_probability_distributions(self):
        """
        Verify that observational end experimental probabilities are a valid 
        distribution.

        Raises:
            ValueError: error when the sum of probabilities !=1.

        Returns:
            None.

        """
        ind_obs_probs = [self.p_obs.get((x, y), 0) for x, y in self.p_obs.keys()]

        if np.round(sum(ind_obs_probs), 1)!=1.0:
            raise ValueError(f'Observational probabilities are a not valid distribution. Sum = {sum(ind_obs_probs)}')
        
        # The marginal probability of each treatment must sum to 1
        for x in self.x_values:
            marg_prob = sum(self.p_exp.get((y, x), 0) for y in self.y_values)
            
            if np.round(marg_prob, 1)!=1.0:
                raise ValueError(f'Marginal probability of treatment {x} is a not valid distribution. Sum = {marg_prob}')
    
    def check_data_completeness(self):
        """
        Check for missing combinations of treatments and effects in the data.
        
        Prints warnings about missing combinations.
        
        Returns:
            None.
        """
        print("\n=== Data Completeness Check ===")
        
        # Check for missing observational probabilities P(x, y)
        missing_obs = []
        for x in self.x_values:
            for y in self.y_values:
                if (x, y) not in self.p_obs:
                    missing_obs.append((x, y))
        
        if missing_obs:
            print(f"\nWARNING: Missing observational probabilities P(x, y):")
            for x, y in missing_obs[:10]:  # Show first 10
                print(f"  - P({x}, {y}) = 0 (missing)")
            if len(missing_obs) > 10:
                print(f"  ... and {len(missing_obs) - 10} more")
        else:
            print("\n✓ All observational probabilities P(x, y) are present")
        
        # Check for missing experimental probabilities P(y|do(x))
        missing_exp = []
        for x in self.x_values:
            for y in self.y_values:
                if (y, x) not in self.p_exp:
                    missing_exp.append((y, x))
        
        if missing_exp:
            print(f"\nWARNING: Missing experimental probabilities P(y|do(x)):")
            for y, x in missing_exp[:10]:  # Show first 10
                print(f"  - P({y}|do({x})) = 0 (missing)")
            if len(missing_exp) > 10:
                print(f"  ... and {len(missing_exp) - 10} more")
        else:
            print("\n✓ All experimental probabilities P(y|do(x)) are present")
        
        # Check for potential issues with marginal probabilities
        print("\n=== Marginal Probability Check ===")
        for x in self.x_values:
            p_x = self.get_p_obs_x(x)
            print(f"P({x}) = {p_x:.4f}")
        
        for y in self.y_values:
            p_y = self.get_p_obs_y(y)
            print(f"P({y}) = {p_y:.4f}")
        
        print("=== End of Data Completeness Check ===\n")
    
    def validate_bounds(self, lb, ub, query=None):
        """
        Validate that bounds are consistent (lb <= ub).
        
        Args:
            lb (float): lower bound
            ub (float): upper bound
            query (dict, optional): the query that produced these bounds
            
        Returns:
            bool: True if bounds are valid, False otherwise
            
        Prints warnings if bounds are inconsistent.
        """
        if lb > ub:
            print(f"\n{'='*60}")
            print(f"ERROR: Inconsistent bounds detected!")
            print(f"{'='*60}")
            print(f"Lower bound: {lb:.6f}")
            print(f"Upper bound: {ub:.6f}")
            print(f"Difference: {lb - ub:.6f}")
            
            if query:
                print(f"\nQuery details:")
                print(f"  Hypotheticals: {query.get('hypotheticals', [])}")
                print(f"  x_p: {query.get('x_p')}")
                print(f"  y_q: {query.get('y_q')}")
            
            print(f"{'='*60}\n")
            return False
        return True
                             
# %% Single Hypothetical Term (Theorems 4-7)


def prob_preservation(y_i, x_j, data):
    """
    Theorem 4: Probability of Preservation P(y_i(x_j), y_i)
    
    The probability that a unit with outcome y_i would remain in 
    outcome y_i if exposed to treatment x_j.

    Args:
        y_i (str): outcome. 
        x_j (str): hypothetical treatment.
        data (Probabilities): observational and experimental probability distributions.
    
    Raises:
        ValueError: Error when y_i or x_j are not values of data.

    Returns:
        lb (float): lower bound.
        ub (float): upper bound.

    """
    if not (y_i in data.y_values):
        raise ValueError(f"Error: unknown outcome {y_i}")
    if not (x_j in data.x_values):
        raise ValueError(f"Error: unknown treatment {x_j}")

    p_exp_yi_xj = data.p_exp.get((y_i, x_j), 0)
    p_obs_xj_yi = data.p_obs.get((x_j, y_i), 0)
    p_obs_yi = data.get_p_obs_y(y_i)

    # Lower bound: max{P(x_j, y_i), P(y_i(x_j)) + P(y_i) - 1}
    lb = max(p_obs_xj_yi, p_exp_yi_xj + p_obs_yi - 1)

    # Upper bound: min{P(y_i(x_j)), P(y_i)}
    ub = min(p_exp_yi_xj, p_obs_yi)

    return lb, ub


def prob_replacement(y_i, x_j, y_k, data):
    """
    Theorem 5: Probability of Replacement P(y_i(x_j), y_k) where i != k
    
    The probability that a unit with outcome y_k would switch to outcome y_i 
    if exposed to treatment x_j.
    
    Args:
        y_i (str): hypothetical outcome.
        x_j (str): hypothetrical treatment.
        y_k (str): outcome.
        data (Probabilities): observational and experimental probability distributions.

    Raises:
        ValueError: Error when y_i, x_j, or y_k are not values of data.
        ValueError: Error when y_i is equal to y_k.

    Returns:
        lb (float): lower bound.
        ub (float): upper bound.

    """

    if not (y_i in data.y_values):
        raise ValueError(f"Error: unknown hypothetical outcome {y_i}")
    if not (y_k in data.y_values):
        raise ValueError(f"Error: unknown outcome {y_k}")
    if not (x_j in data.x_values):
        raise ValueError(f"Error: unknown hypothetical treatment {x_j}")

    if y_i == y_k:
        raise ValueError(
            "For Probability of Replacement, y_i must not equal y_k.")

    p_exp_yi_xj = data.p_exp.get((y_i, x_j), 0)  # P(y_i(x_j))
    p_obs_yk = data.get_p_obs_y(y_k)  # P(y_k)
    p_obs_xj_yi = data.p_obs.get((x_j, y_i), 0)  # P(x_j, y_i)
    p_obs_yk_xj = data.p_obs.get((x_j, y_k), 0)  # P(y_k, x_j)
    p_obs_xj = data.get_p_obs_x(x_j)  # P(x_j)

    # LOWER BOUND
    # The lower bound is computed with a max function. Its third argument is
    # a sum over a max. This sum iterates over p for p!=j, where m is the
    # number/index of possible x values.

    # From m (data.x_values), determine the remaining p such that p != j:
    p = list(set(data.x_values).difference(x_j))

    # Terms of the max function inside the third argument:
    max_terms = []

   # Iterate over p for p!=j:
    for x_p in p:
        p_obs_yk_xp = data.p_obs.get((x_p, y_k), 0)
        max_terms.append(max(0, p_exp_yi_xj + p_obs_yk_xp -
                         1 + p_obs_xj - p_obs_xj_yi))

    lb = max(0, p_exp_yi_xj + p_obs_yk - 1, sum(max_terms))

    # UPPER BOUND
    ub = min(p_exp_yi_xj - p_obs_xj_yi, p_obs_yk - p_obs_yk_xj)

    return lb, ub


def prob_substitute(y_i, x_j, x_k, data):
    """
    Theorem 6: Probability of Substitute P(y_i(x_j), x_k) where j != k
    
    The probability that a unit exposed to treatment x_k would have outcome y_i
    had it been exposed to treatment x_j instead.
    

    Args:
        y_i (TYPE): hypothetical outcome.
        x_j (TYPE): hypothetical treatment.
        x_k (TYPE): treatment.
        data (Probabilities): observational and experimental probability distributions.

    Raises:
        ValueError: Error when y_i, x_j, or x_k are not values of data.
        ValueError: Error when x_j is equal to x_k.

    Returns:
        lb (float): lower bound.
        ub (float): upper bound.

    """

    if not (y_i in data.y_values):
        raise ValueError(f"Error: unknown hypothetical outcome {y_i}")
    if not (x_j in data.x_values):
        raise ValueError(f"Error: unknown hypothetical treatment {x_j}")
    if not (x_k in data.x_values):
        raise ValueError(f"Error: unknown treatment {x_k}")

    if x_j == x_k:
        raise ValueError(
            "For Probability of Substitute, x_j must not equal x_k.")

    p_exp_yi_xj = data.p_exp.get((y_i, x_j), 0)  # P(y_i(x_j))
    p_obs_xj_yi = data.p_obs.get((x_j, y_i), 0)  # P(x_j, y_i)
    p_obs_xj = data.get_p_obs_x(x_j)  # P(x_j)
    p_obs_xk = data.get_p_obs_x(x_k)  # P(x_k)

    lb = max(0, p_exp_yi_xj - p_obs_xj_yi - 1 + p_obs_xj + p_obs_xk)
    ub = min(p_exp_yi_xj - p_obs_xj_yi, p_obs_xk)

    return lb, ub


def prob_necessity(y_i, x_j, y_k, x_p, data):
    """
    Theorem 7: Probability of Necessity P(y_i(x_j), y_k, x_p) where j != p
    
    The probability that a unit with treatment x_p and outcome y_k
    would have had outcome y_i if it had received treatment x_j.

    Args:
        y_i (str): hypothetical outcome.
        x_j (str): hypothetical treatment.
        y_k (str): outcome.
        x_p (str): treatment.
        data (Probabilities): observational and experimental probability distributions.


    Raises:
        ValueError: Error when y_i, x_j, y_k, or x_k are not values of data.
        ValueError: Error when x_j is equal to x_p.


    Returns:
        lb (float): lower bound.
        ub (float): upper bound.

    """

    if not (y_i in data.y_values):
        raise ValueError(f"Error: unknown hypothetical outcome {y_i}")
    if not (x_j in data.x_values):
        raise ValueError(f"Error: unknown hypothetical treatment {x_j}")
    if not (y_k in data.y_values):
        raise ValueError(f"Error: unknown outcome {y_k}")
    if not (x_p in data.x_values):
        raise ValueError(f"Error: unknown treatment {x_p}")

    if x_j == x_p:
        raise ValueError(
            "For Probability of Necessity, x_j must not equal x_p.")

    p_exp_yi_xj = data.p_exp.get((y_i, x_j), 0)  # P(y_i(x_j))
    p_obs_xp_yk = data.p_obs.get((x_p, y_k), 0)  # P(x_p, y_k)
    p_obs_xj = data.get_p_obs_x(x_j)  # P(x_j)
    p_obs_xj_yi = data.p_obs.get((x_j, y_i), 0)  # P(x_j, y_i)

    lb = max(0, p_exp_yi_xj + p_obs_xp_yk - 1 + p_obs_xj - p_obs_xj_yi)
    ub = min(p_exp_yi_xj - p_obs_xj_yi, p_obs_xp_yk)

    return lb, ub

# %% Multiple Hypothetical Terms (Theorems 8-11)


def prob_necessity_sufficiency_k(hypotheticals, data):
    """
    Theorem 8: Probability of Necessity and Sufficiency with k hypothetical terms
    P(y_i1(x_j1), ..., y_ik(x_jk))

    Args:
        hypotheticals (List): list of tuples with hypothetical terms (y, x).
        data (Probabilities): observational and experimental probability distributions.

    Raises:
        ValueError: Error when hypothetical terms are not values of data.
    
    Returns:
        theorem_8_lb (float): lower bound.
        theorem_8_ub (float): upper bound.

    """

    for y, x in hypotheticals:
        if not (y in data.y_values):
            raise ValueError(f"Error: unknown hypothetical outcome {y}")
        if not (x in data.x_values):
            raise ValueError(f"Error: unknown hypothetical treatment {x}")

    k = len(hypotheticals)

    if k == 0:
        raise ValueError("Error: no hypothetical terms")
    if k == 1:
        y, x = hypotheticals[0]
        p = data.p_exp.get((y, x), 0)
        return (p, p)

    if k >= 2:
        #######################################################################
        # Recursive lower and upper bounds of the joint probability of the
        # hypothethical terms (a.k.a probability of necessity and sufficiency - theorem 8)
        # for the third argument of the lower bound and the second argument
        # of the upper bound
        necessity_sufficiency_recursive_lbs = []
        necessity_sufficiency_recursive_ubs = []

        for t in range(k):
            yit, xit = hypotheticals[t]
            p_exp_yit_xit = data.p_exp.get((yit, xit), 0)  # P(y_it(x_it))

            # remaining hypothethicals
            remaining = hypotheticals[:t] + hypotheticals[t+1:]

            if len(remaining) == 1:
                y, x = remaining[0]
                p = data.p_exp.get((y, x), 0)
                necessity_sufficiency_recursive_lbs.append(p+p_exp_yit_xit-1)
                necessity_sufficiency_recursive_ubs.append(p)
            elif len(remaining) >= 2:
                # bounds of two or more hypothetical terms
                # recursive application of theorem 8
                lb, ub = prob_necessity_sufficiency_k(remaining, data)

                necessity_sufficiency_recursive_lbs.append(lb+p_exp_yit_xit-1)
                necessity_sufficiency_recursive_ubs.append(ub)
            else:
                raise ValueError("Error in recursive application of theorem 8")

        #######################################################################
        # Recursive lower and upper bounds of the probability of necessity (Theorem 7 or 11)
        # for the fourth argument (first summand) of the lower bound
        # and the third argument (first summand) of the upper bound

        necessity_recursive_lbs = []
        necessity_recursive_ubs = []

        for r in range(k):
            remaining = hypotheticals[:r] + hypotheticals[r+1:]

            if len(remaining) == 1:
                # apply theorem 7
                # the name of the retrieved arguments follows the
                # names of the prob_necessity function
                y_i, x_j = remaining[0]
                y_k, x_p = hypotheticals[r]

                lb, ub = prob_necessity(y_i, x_j, y_k, x_p, data)

                necessity_recursive_lbs.append(lb)
                necessity_recursive_ubs.append(ub)

            elif len(remaining) >= 2:
                # bounds of two or more hypothetical terms
                # recursive application of theorem 11
                y_k, x_p = hypotheticals[r]
                lb, ub = prob_necessity_k_p_q(remaining, x_p, y_k, data)

                necessity_recursive_lbs.append(lb)
                necessity_recursive_ubs.append(ub)

            else:
                raise ValueError(
                    "Error in recursive application of theorems 7 and 11 ")

        #######################################################################
        # Recursive lower and upper bounds of the probability of substitute (Theorem 6 or 9)
        # for the fourth argument (second summand) of the lower bound
        # and the third argument (second summand) of the upper bound

        # m is the mumber of possible x values
        # determine the remaining p such that p != j1 ... != jk
        hypothetical_x = list(term[1] for term in hypotheticals)

        p = list(set(data.x_values).difference(hypothetical_x))

        # recursive probability of substitute
        substitute_recursive_lbs = []
        substitute_recursive_ubs = []

        for x_p in p:
            if k == 1:
                # apply theorem 6 (probability of substitute with one hypothetical term)
                y_i, x_j = hypotheticals[0]
                x_k = x_p
                lb, ub = prob_substitute(y_i, x_j, x_k, data)
            elif k >= 2:
                # apply theorem 9 (probability of substitute for two or more hypothetical terms)
                lb, ub = prob_substitute_k_p(hypotheticals, x_p, data)

            substitute_recursive_lbs.append(lb)
            substitute_recursive_ubs.append(ub)

        #######################################################################
        # Probabilities for second argument of lower bound and first argument
        # of upper bound
        individual_probs = [data.p_exp.get((y, x), 0)
                            for y, x in hypotheticals]

        #######################################################################
        # Finally, determine the upper and lower bounds

        theorem_8_lb = max(0,
                           sum(individual_probs)-k+1,
                           max(necessity_sufficiency_recursive_lbs),
                           sum(necessity_recursive_lbs) + sum(substitute_recursive_lbs))

        theorem_8_ub = min(min(individual_probs),
                           min(necessity_sufficiency_recursive_ubs),
                           sum(necessity_recursive_ubs) + sum(substitute_recursive_ubs))

        return theorem_8_lb, theorem_8_ub


def prob_substitute_k_p(hypotheticals, x_p, data):
    """
    Theorem 9: Probability of Substitute with k hypothetical terms and treatment x_p
    P(y_i1(x_j1), ..., y_ik(x_jk), x_p)

    Args:
        hypotheticals (List): list of tuples with hypothetical terms (y, x).
        x_p (str): treatment.
        data (Probabilities): observational and experimental probability distributions.

    Raises:
        ValueError: Error when hypothetical terms are not values of data.
        ValueError: Error when treatment is not value of data.

    Returns:
        theorem_9_lb (float): lower bound.
        theorem_9_ub (float): upper bound.

    """

    for y, x in hypotheticals:
        if not (y in data.y_values):
            raise ValueError(f"Error: unknown hypothetical outcome {y}")
        if not (x in data.x_values):
            raise ValueError(f"Error: unknown hypothetical treatment {x}")

    if not (x_p in data.x_values):
        raise ValueError(f"Error: unknown treatment {x_p}")

    k = len(hypotheticals)

    if k == 0:
        raise ValueError("Error: no hypothetical terms")

    if k == 1:
        y, x = hypotheticals[0]
        lb, ub = prob_substitute(y, x, x_p, data)

        return lb, ub

    ###########################################################################
    # Probabilities for second argument of lower bound and first argument
    # of upper bound
    individual_probs = [data.p_exp.get((y, x), 0) for y, x in hypotheticals]

    ###########################################################################
    # Probability for second argument of lower bound and the second argument
    # of the upper bound
    p_xp = data.get_p_obs_x(x_p)

    ###########################################################################
    # Recursive lower and upper bounds of the probability of necessity and
    # sufficiency (theorem 8)
    # for the third argument (first summand) of the lower bound and the third
    # argument of the upper bound
    pns_recursive_lbs = []
    pns_recursive_ubs = []

    # Recursive lower and upper bounds of the probability of substitute (Theorem 6)
    # for the third argument (second summand) of the lower bound
    # and the fourth argument of the upper bound
    ps_recursive_lbs = []
    ps_recursive_ubs = []

    for t in range(k):
        remaining = hypotheticals[:t] + hypotheticals[t+1:]
        if len(remaining) == 1:
            y, x = remaining[0]
            p = data.p_exp.get((y, x), 0)
            pns_recursive_lbs.append(p)
            pns_recursive_ubs.append(p)

        elif len(remaining) >= 2:
            # apply theorem 8
            lb, ub = prob_necessity_sufficiency_k(remaining, data)
            pns_recursive_lbs.append(lb)
            pns_recursive_ubs.append(ub)

        y_i, x_j = hypotheticals[t]
        x_k = x_p
        lb, ub = prob_substitute(y_i, x_j, x_k, data)

        ps_recursive_lbs.append(lb)
        ps_recursive_ubs.append(ub)

    theorem9_lb = max(0,
                      sum(pns_recursive_lbs) + p_xp - k,
                      np.max(np.array(pns_recursive_lbs) + np.array(ps_recursive_lbs) - 1))

    theorem9_ub = min(min(individual_probs),
                      p_xp,
                      min(pns_recursive_ubs),
                      min(ps_recursive_ubs))

    return theorem9_lb, theorem9_ub


def prob_replacement_k_q(hypotheticals, y_q, data):
    """
    Theorem 10: Probability of replacement with k hypothetical terms and outcome y_q
    
    P(y_i1(x_j1), ..., y_ik(x_jk), y_p)
    
    Args:
        hypotheticals (List): list of tuples with hypothetical terms (y, x).
        y_q (str): outcome.
        data (Probabilities): observational and experimental probability distributions.

    Raises:
        ValueError: Error when hypothetical terms are not values of data.
        ValueError: Error when outcome is not value of data.

    Returns:
        theorem10_lb (float): lower bound.
        theorem10_ub (float): upper bound.

    """

    for y, x in hypotheticals:
        if not (y in data.y_values):
            raise ValueError(f"Error: unknown hypothetical outcome {y}")
        if not (x in data.x_values):
            raise ValueError(f"Error: unknown hypothetical treatment {x}")

    if not (y_q in data.y_values):
        raise ValueError(f"Error: unknown outcome {y_q}")

    k = len(hypotheticals)

    if k == 0:
        raise ValueError("Error: no hypothetical terms")

    ###########################################################################
    # Experimemtal probabilities for second argument of the lower bound and
    # the first argument of the upper bound
    individual_probs = [data.p_exp.get((y, x), 0) for y, x in hypotheticals]

    ###########################################################################
    # Observational probability for the second argument of the lower bound and
    # the second argument of the upper bound
    p_y_q = data.get_p_obs_y(y_q)

    ###########################################################################
    # Recursive lower and upper bounds of the probability of necessity and
    # sufficiency (theorem 8)
    # for the third argument (first summand) of the lower bound and the third
    # argument of the upper bound
    pns_recursive_lbs = []
    pns_recursive_ubs = []

    # Recursive lower and upper bounds of the probability of preservation (theorem 4)
    # or probability of replacement (theorem 5)
    ppr_recursive_lbs = []
    ppr_recursive_ubs = []

    for t in range(k):
        remaining = hypotheticals[:t] + hypotheticals[t+1:]

        if len(remaining) == 1:
            y, x = remaining[0]
            p = data.p_exp.get((y, x), 0)
            pns_recursive_lbs.append(p)
            pns_recursive_ubs.append(p)

        elif len(remaining) >= 2:
            # apply theorem 8
            lb, ub = prob_necessity_sufficiency_k(remaining, data)
            pns_recursive_lbs.append(lb)
            pns_recursive_ubs.append(ub)

        # Apply theorem 4 or 5
        # theorem 4 when P(yi_xj, yi) , i.e., the effects are the same
        # theorem 5 when P(yi_xj, yk), i.e., the effects are different

        y_it, x_it = hypotheticals[t]
        if y_it == y_q:
            # theorem 4 when P(yi_xj, yi) , i.e., the effects are the same
            # P(y_i(x_j), y_i)
            lb, ub = prob_preservation(y_it, x_it, data)
            ppr_recursive_lbs.append(lb)
            ppr_recursive_ubs.append(ub)
        else:
            # theorem 5 when P(yi_xj, yk), i.e., the effects are different
            # P(y_i(x_j), y_k)
            lb, ub = prob_replacement(y_it, x_it, y_q, data)
            ppr_recursive_lbs.append(lb)
            ppr_recursive_ubs.append(ub)

    #####################################################################################
    # Recursive lower and upper bounds of the probability of necessity (Theorem 7 or 11)
    # First term of the sum
    necessity_recursive_lbs = []
    necessity_recursive_ubs = []

    for r in range(k):
        remaining = hypotheticals[:r] + hypotheticals[r+1:]

        if len(remaining) == 1:
            # apply theorem 7
            # the name of the retrieved arguments follows the
            # names of the prob_necessity function
            y_i, x_j = remaining[0]
            y_k, x_p = hypotheticals[r]
            if y_q!=y_k:
                continue
            lb, ub = prob_necessity(y_i, x_j, y_k, x_p, data)
            necessity_recursive_lbs.append(lb)
            necessity_recursive_ubs.append(ub)

        elif len(remaining) >= 2:
            # apply theorem 11
            y_k, x_p = hypotheticals[r]
            if y_q!=y_k:
                continue
            lb, ub = prob_necessity_k_p_q(remaining, x_p, y_k, data)
            necessity_recursive_lbs.append(lb)
            necessity_recursive_ubs.append(ub)

    sum_necessity_recursive_lbs = sum(necessity_recursive_lbs)
    sum_necessity_recursive_ubs = sum(necessity_recursive_ubs)

    ####################################################################################
    # Recursive lower and upper bounds of the probability of necessity (Theorem 7 or 11)
    # Second term of the sum

    # m is the number of possible x values
    # determine the remaining p such that p != j1 ... != jk

    hypothetical_x = list(term[1] for term in hypotheticals)

    p = list(set(data.x_values).difference(hypothetical_x))

    # recursive probability of substitute
    second_necessity_recursive_lbs = []
    second_necessity_recursive_ubs = []

    for x_p in p:
        if k == 1:
            # apply theorem 7 (probability of necessity with one hypothetical term)
            y_i, x_j = hypotheticals[0]
            lb, ub = prob_necessity(y_i, x_j, y_q, x_p, data)
            second_necessity_recursive_lbs.append(lb)
            second_necessity_recursive_ubs.append(ub)

        elif k >= 2:
            # apply theorem 11
            lb, ub = prob_necessity_k_p_q(hypotheticals, x_p, y_q, data)
            second_necessity_recursive_lbs.append(lb)
            second_necessity_recursive_ubs.append(ub)

    sum_second_necessity_recursive_lbs = sum(second_necessity_recursive_lbs)
    sum_second_necessity_recursive_ubs = sum(second_necessity_recursive_ubs)

    ##############################################################
    # Finally, determine the upper and lower bounds

    theorem10_lb = max(0,
                       sum(individual_probs) + p_y_q - k,
                       max(np.array(pns_recursive_lbs) +
                           np.array(ppr_recursive_lbs) - 1),
                       sum_necessity_recursive_lbs + sum_second_necessity_recursive_lbs)

    theorem10_ub = min(min(individual_probs),
                       p_y_q,
                       min(pns_recursive_ubs),
                       min(ppr_recursive_ubs),
                       sum_necessity_recursive_ubs + sum_second_necessity_recursive_ubs)

    return theorem10_lb, theorem10_ub


def prob_necessity_k_p_q(hypotheticals, x_p, y_q, data):
    """
    Theorem 11: Probability of necessity with k hypothetical terms, treatment x_p 
    and outcome y_q  
    
    P(y_i1(x_j1), ..., y_ik(x_jk), x_p, y_q)
    
    Args:
        hypotheticals (List): list of tuples with hypothetical terms (y, x).
        x_p (str): treatment.
        y_q (TYPE): outcome.
        data (Probabilities): observational and experimental probability distributions.

    Raises:
        ValueError: Error when hypothetical terms are not values of data.
        ValueError: Error when treatment or outcome are not values of data.

    Returns:
        theorem11_lb (float): lower bound.
        theorem11_ub (float): upper bound.

    """
    for y, x in hypotheticals:
        if not (y in data.y_values):
            raise ValueError(f"Error: unknown hypothetical outcome {y}")
        if not (x in data.x_values):
            raise ValueError(f"Error: unknown hypothetical treatment {x}")

    if not (x_p in data.x_values):
        raise ValueError(f"Error: unknown treatment {x_p}")

    if not (y_q in data.y_values):
        raise ValueError(f"Error: unknown outcome {y_q}")

    k = len(hypotheticals)

    if k == 0:
        raise ValueError("Error: no hypothetical terms")

    if k == 1:
        y_i, x_j = hypotheticals[0]
        lb, ub = prob_necessity(y_i, x_j, y_q, x_p, data)
        return (lb, ub)

    ###########################################################################
    # Experimemtal probabilities for second argument of the lower bound and
    # the first argument of the upper bound
    individual_probs = [data.p_exp.get((y, x), 0) for y, x in hypotheticals]

    ###########################################################################
    # Observational probability for second argument of the lower bound and
    # the second argument of the upper bound
    p_xp_yq = data.p_obs.get((x_p, y_q), 0)

    ###########################################################################
    # Recursive lower and upper bounds of the probability of necessity and
    # sufficiency (theorem 8)
    # for the third argument (first summand) of the lower bound and the third
    # argument of the upper bound
    pns_recursive_lbs = []
    pns_recursive_ubs = []

    # Recursive lower and upper bounds of the probability of necessity (Theorem 7)
    # for the third argument (second summand) of the lower bound and the fourth
    # argument of the upper bound
    pn_recursive_lbs = []
    pn_recursive_ubs = []

    for t in range(k):
        remaining = hypotheticals[:t] + hypotheticals[t+1:]
        if len(remaining) == 1:
            y, x = remaining[0]
            p = data.p_exp.get((y, x), 0)
            pns_recursive_lbs.append(p)
            pns_recursive_ubs.append(p)

        elif len(remaining) >= 2:
            # apply theorem 8
            lb, ub = prob_necessity_sufficiency_k(remaining, data)
            pns_recursive_lbs.append(lb)
            pns_recursive_ubs.append(ub)

        # Apply theorem 7
        y_i, x_j = hypotheticals[t]
        lb, ub = prob_necessity(y_i, x_j, y_q, x_p, data)
        pn_recursive_lbs.append(lb)
        pn_recursive_ubs.append(ub)

    theorem11_lb = max(0,
                       sum(individual_probs) + p_xp_yq - k,
                       np.max(np.array(pns_recursive_lbs) +
                              np.array(pn_recursive_lbs) - 1)
                       )
    theorem11_ub = min(min(individual_probs),
                       p_xp_yq,
                       min(pns_recursive_ubs),
                       min(pn_recursive_ubs)
                       )

    return theorem11_lb, theorem11_ub

# %% Dispatcher function


def get_bounds(query, data):
    """
    Dispatcher function to call a theorem according to the query.

    Args:
        query (Dict): query with the keys 'hypotheticals', 'x_p', and 'y_q'.
        data (Probabilities): observational and experimental probability distributions.

    Returns:
        float: lower bound.
        float: upper bound.

    """
    hypotheticals = query.get("hypotheticals", [])
    x_p = query.get("x_p")
    y_q = query.get("y_q")

    hypotheticals = [tuple(h) for h in hypotheticals]
    k = len(hypotheticals)

    # Single hypothetical term cases (Theorems 4-7)
    if k == 1:
        y, x = hypotheticals[0]

        if x_p and y_q:
            # theorem 7
            print('Applying theorem 7: probability of necessity')
            lb, ub = prob_necessity(y, x, y_q, x_p, data)
        elif x_p:
            # theorem 6
            print('Applying theorem 6: probability of substitute')
            lb, ub = prob_substitute(y, x, x_p, data)
        elif y_q:
            if y == y_q:
                # theorem 4
                print('Applying theorem 4: probability of preservation')
                lb, ub = prob_preservation(y, x, data)
            else:
                # theorem 5
                print('Applying theorem 5: probability of replacement')
                lb, ub = prob_replacement(y, x, y_q, data)
        else:
            p = data.p_exp.get((y, x), 0)
            return (p, p)

    # Multi-hypothetical term cases (Theorems 8-11)
    elif k >= 2:
        if x_p and y_q:
            print('Applying theorem 11: probability of necessity')
            lb, ub = prob_necessity_k_p_q(hypotheticals, x_p, y_q, data)
        elif x_p:
            print('Applying theorem 9: probability of substitute')
            lb, ub = prob_substitute_k_p(hypotheticals, x_p, data)
        elif y_q:
            print('Applying theorem 10: probability of replacement')
            lb, ub = prob_replacement_k_q(hypotheticals, y_q, data)
        else:
            print('Applying theorem 8: probability of necessity and sufficiency')
            lb, ub = prob_necessity_sufficiency_k(hypotheticals, data)

    # Base cases for no hypothetical terms
    else:
        if x_p and y_q:
            p = data.p_obs.get((x_p, y_q), 0)
            return (p, p)
        elif x_p:
            p = data.get_p_obs_x(x_p)
            return (p, p)
        elif y_q:
            p = data.get_p_obs_y(y_q)
            return (p, p)
        else:
            return (np.nan, np.nan)
    
    # Validate bounds before returning
    data.validate_bounds(lb, ub, query)
    
    return lb, ub
