import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import itertools



UPPER_LIMIT = 1000000
LOWER_LIMIT = -1000000

class DecisionTableClassifier:
 
    def __init__(self, max_depth=5, max_leaf_nodes=16, n_lines=16, client_id=0, n_clients=1, left_trees=None, own_tree_influence=None):
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.final_leaf_nodes = n_lines
        self.client_id = client_id
        self.n_clients = n_clients

        if self.final_leaf_nodes < self.max_leaf_nodes:
            self.final_leaf_nodes = self.max_leaf_nodes

        if left_trees != None:
            self.left_trees = left_trees
        else:
            self.left_trees = n_clients
        
        if own_tree_influence != None:
            self.own_tree_influence = own_tree_influence
        else:   # simple majority rule
            self.own_tree_influence = 1

        self.current_decision_table = []
        self.current_value_table = []


    def fit(self, X, Y):
        ######## create DTmodel ###########
        self.DTmodel = self.createDtModel()
        self.DTmodel.fit(X, Y)
        ###################################

        ######## get DTmodel information #########
        self.left_node_list,self.right_node_list,self.feature_list,self.threshold_list,self.n_node_samples_list,self.label_list,self.labeling_value_list = self.getDtModelInfo(self.DTmodel)
        self.n_features = len(X[1])
        self.n_nodes = len(self.feature_list)
        self.n_classes = int(max(self.label_list)) + 1
        ###########################################

        ######## get leaf node information #########
        self.leaf_node_index_list, self.leaf_node_class_list, self.leaf_node_value_list, self.pass_list = self.getLeafNodeInfo()
        ############################################

        ######################### create(fit) decision table ####################################
        self.sent_decision_table, self.sent_value_table = self.createDecisionTable()
        #########################################################################################



    def createDtModel(self):
        model = DecisionTreeClassifier(max_depth=self.max_depth, max_leaf_nodes=self.max_leaf_nodes, random_state=0)
        return model


    def getDtModelInfo(self, model):
        left_node_list = model.tree_.children_left  # the node connected each node leftside(leaf node:-1)
        right_node_list = model.tree_.children_right # the node connected each node rightside(leaf node:-1)
        feature_list = model.tree_.feature # the feature used by each node(leaf node:-2)
        threshold_list = model.tree_.threshold # the threshold used by each node(leaf node:-2)
        n_node_samples_list = model.tree_.n_node_samples # the number of samples in each node
        
        values_list = model.tree_.value
        flatten_values_list = []
        for l in values_list:
            n_labels = len(list(itertools.chain.from_iterable(l)))
            flatten_values_list.append(list(itertools.chain.from_iterable(l)))

        label_list = []
        for k in flatten_values_list:
            label_list.append(np.argmax(k))

        labeling_value_list = []
        for m in flatten_values_list:
            sum = 0
            for v in m:
                sum += int(v)
            value_list = []
            for v in m:
                value_list.append(float(v)/sum)
            labeling_value_list.append(value_list)
        
        return left_node_list, right_node_list, feature_list, threshold_list, n_node_samples_list, label_list, labeling_value_list


    def getNumberOfLeafNodeSamples(self, feature_list, n_node_samples_list):
        n_leaf_node_samples = 0
        for i in range(len(feature_list)):
            if int(feature_list[i]) == -2:  # if leaf node
                n_leaf_node_samples += n_node_samples_list[i]

        return n_leaf_node_samples


    def getLeafNodeInfo(self):
        leaf_node_index_list = []   # list of leaf node index
        leaf_node_class_list = []   # list of leaf node class
        leaf_node_value_list = []   # list of leaf node value 
        pass_list = []              # list of path (leaf to root)

        for i in range(len(self.feature_list)):
            if int(self.feature_list[i]) == -2: # if leaf node
                leaf_node_index_list.append(i)
                leaf_node_class_list.append(self.label_list[i])
                leaf_node_value_list.append(self.labeling_value_list[i])

        for leaf_index in leaf_node_index_list:
            pass_list.append(self.getLeafToRootPass(leaf_index))

        return leaf_node_index_list, leaf_node_class_list, leaf_node_value_list, pass_list


    def getLeafToRootPass(self, leaf_node_index):
        current_index = leaf_node_index

        index_list = []
        index_list.append(leaf_node_index)

        while(1):
            current_index = self.getParentNode(current_index)
            index_list.append(current_index)
            if current_index == 0:  # if root node
                break

        return index_list


    def getParentNode(self, child_node_index):
        for i in range(len(self.left_node_list)):
            if int(self.left_node_list[i]) == int(child_node_index):
                return i
        for l in range(len(self.right_node_list)):
            if int(self.right_node_list[l]) == int(child_node_index):
                return l

        return 0     # root node


    def createDecisionTable(self):
        
        ############################# create origin decision table ####################################
        origin_decision_table = []
        for ps in self.pass_list:   #loop(each pass)
            added_list = [LOWER_LIMIT, UPPER_LIMIT] * self.n_features    # initialize added list

            for node_index in ps:      #loop(each node in pass)
                if node_index == 0:    # root node
                    pass
                else:   # other node
                    p_node_index = self.getParentNode(node_index)   # get parent node index
                    p_node_feature = self.feature_list[p_node_index]    # get parent node feature
                    p_node_threshold = self.threshold_list[p_node_index]    # get parent node threshold

                    if node_index in self.left_node_list:  # value is less than threshold
                        if added_list[p_node_feature*2+1] == UPPER_LIMIT:
                            added_list[p_node_feature*2+1] = p_node_threshold

                    elif node_index in self.right_node_list:   #value is more than threshold
                        if added_list[p_node_feature*2] == LOWER_LIMIT:
                            added_list[p_node_feature*2] = p_node_threshold

            origin_decision_table.append(added_list)
        ###############################################################################################

        complemented_decision_table, complemented_leaf_node_value_list = self.complementDecisionTable(origin_decision_table, self.leaf_node_value_list)
        inserted_decision_table, inserted_leaf_node_value_list = self.insertDecisionTable(complemented_decision_table, complemented_leaf_node_value_list)

        returned_decision_table = inserted_decision_table
        returned_leaf_node_value_list = inserted_leaf_node_value_list

        return returned_decision_table, returned_leaf_node_value_list


    def complementDecisionTable(self, origin_decision_table, leaf_node_value_list):
        n_decision_table_rows = self.final_leaf_nodes

        returned_decision_table = origin_decision_table
        returned_leaf_node_value_list = leaf_node_value_list

        if len(returned_decision_table) < n_decision_table_rows:
            complemented_count = n_decision_table_rows - len(returned_decision_table)

            for i in range(complemented_count):
                returned_decision_table.append(returned_decision_table[i])
                returned_leaf_node_value_list.append(returned_leaf_node_value_list[i])

        return returned_decision_table, returned_leaf_node_value_list


    def insertDecisionTable(self, decision_table, leaf_node_value_list):
        returned_decision_table = []
        returned_value_table = []

        added_decision_list = [0.0, 0.0] * self.n_features
        added_decision_table = []
        for i in range(self.final_leaf_nodes):
            added_decision_table.append(added_decision_list)

        added_value_list = [0] * self.n_classes
        added_value_table = []
        for l in range(self.final_leaf_nodes):
            added_value_table.append(added_value_list)

        for cnt in range(self.n_clients):
            if cnt == self.client_id:
                returned_decision_table += decision_table
                returned_value_table += leaf_node_value_list
            else:
                returned_decision_table += added_decision_table
                returned_value_table += added_value_table

        return returned_decision_table, returned_value_table


    def thin_trees(self, left_trees=None):
        if left_trees != None:
            self.left_trees = left_trees

        current_n_trees = len(self.current_decision_table) / self.final_leaf_nodes
        if current_n_trees > self.left_trees:
            tmp1_decision_table = self.current_decision_table
            tmp1_value_table = self.current_value_table

            tmp1_decision_table.reverse()
            tmp1_value_table.reverse()

            n_left_leaf_nodes = self.left_trees * self.final_leaf_nodes

            tmp2_decision_table = tmp1_decision_table[0:n_left_leaf_nodes]
            tmp2_value_table = tmp1_value_table[0:n_left_leaf_nodes]

            tmp2_decision_table.reverse()
            tmp2_value_table.reverse()

            self.current_decision_table = tmp2_decision_table
            self.current_value_table = tmp2_value_table
        else:
            pass


    def predict(self, X):
        Y_pred = []

        decision_table = np.array(self.current_decision_table)
        value_table = np.array(self.current_value_table)

        other_tree_influence = 1
        for x in X: # x:1 data   X:predicted data overall

            pred_vote_list = [0] * self.n_classes

            match_score_list = []
            tree_count = 0
            for i in range(len(decision_table)): # loop(rows in decision table)

                match_score = 0
                match_count = 0
                for l in range(int(len(decision_table[i])/2)):    #loop(features)
                    if (x[l] >= decision_table[i][l*2]) and (x[l] < decision_table[i][l*2+1]):  # value is in range
                        match_score += 1
                        match_count += 1
                    else:   # value is not in range
                        match_score -= 1
                match_score_list.append(match_score)
                
                if (i+1) % self.final_leaf_nodes == 0:
                    most_match_row_index = match_score_list.index(max(match_score_list))    # get most match row index
                    most_match_row_value_list = value_table[(tree_count * self.final_leaf_nodes) + most_match_row_index]    # get most match row value list
                    pred_vote_class = np.argmax(most_match_row_value_list)  # get pred class

                    if (tree_count % self.n_clients) == self.client_id:
                        pred_vote_list[pred_vote_class] += self.own_tree_influence
                    else:
                        pred_vote_list[pred_vote_class] += other_tree_influence
                    match_score_list = []
                    tree_count += 1

            y_p = pred_vote_list.index(max(pred_vote_list)) # get pred class (final)
            Y_pred.append(int(y_p))

        return Y_pred


    def predict_proba(self, X):
        Y_pred_proba = []

        decision_table = np.array(self.current_decision_table)
        value_table = np.array(self.current_value_table)

        other_tree_influence = 1
        for x in X: # x:1 data   X:predicted data overall

            pred_vote_list = [0] * self.n_classes

            match_score_list = []
            tree_count = 0
            for i in range(len(decision_table)): # loop(rows in decision table)

                match_score = 0
                for l in range(int(len(decision_table[i])/2)):    #loop(features)
                    if (x[l] >= decision_table[i][l*2]) and (x[l] < decision_table[i][l*2+1]):  # value is in range
                        match_score += 1
                    else:   # value is not in range
                        match_score -= 1
                match_score_list.append(match_score)

                if (i+1) % self.final_leaf_nodes == 0:
                    most_match_row_index = match_score_list.index(max(match_score_list))
                    most_match_row_value_list = value_table[(i+1-self.final_leaf_nodes) + most_match_row_index]
                    pred_vote_class = np.argmax(most_match_row_value_list)

                    if (tree_count % self.n_clients) == self.client_id:
                        pred_vote_list[pred_vote_class] += self.own_tree_influence
                    else:
                        pred_vote_list[pred_vote_class] += other_tree_influence
                    match_score_list = []
                    tree_count += 1

            y_p_p = []
            pred_vote_sum = sum(pred_vote_list)
            if pred_vote_sum != 0:
                for cl in pred_vote_list:
                    y_p_p.append(float(cl/pred_vote_sum))
            else:
                y_p_p = pred_vote_list
            Y_pred_proba.append(y_p_p)

        return Y_pred_proba


    def calc_accuracy_score(self, X, Y_true):
        Y_pred = self.predict(X)
        return accuracy_score(Y_true, Y_pred)


    def calc_precision_score(self, X, Y_true):
        Y_pred = self.predict(X)
        return precision_score(Y_true, Y_pred)


    def calc_recall_score(self, X, Y_true):
        Y_pred = self.predict(X)
        return recall_score(Y_true, Y_pred)


    def calc_f1_score(self, X, Y_true):
        Y_pred = self.predict(X)
        return f1_score(Y_true, Y_pred)


    def get_params(self):
        params = [self.sent_decision_table, self.sent_value_table]
        return params


    def set_params(self, parameters):
        received_decision_table = parameters[0].tolist()
        received_value_table = parameters[1].tolist()

        multiplied_decision_table = []
        multiplied_value_table = []

        for i in range(len(received_decision_table)):
            multiplied_decision_table.append([x*self.n_clients for x in received_decision_table[i]])
            multiplied_value_table.append([y*self.n_clients for y in received_value_table[i]])

        self.current_decision_table += multiplied_decision_table
        self.current_value_table += multiplied_value_table


    def plot_tree(self):
        plt.figure(figsize=(15, 15))
        plot_tree(self.DTmodel, filled=True)
        plt.show()


    def get_decision_table_n_lines(self):
        return len(self.current_decision_table)


    def get_n_trees(self):
        return int(len(self.current_decision_table) / self.final_leaf_nodes)


    def print_decision_table(self):
        for i in range(len(self.current_decision_table)):
            print(f"{self.current_decision_table[i]}{self.current_value_table[i]}")
            if (i+1) % self.final_leaf_nodes == 0:
                print("")