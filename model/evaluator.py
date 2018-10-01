
import random

from dbPointer import queryResultVenues
from delexicalize import *
from nlp import *

domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
requestables = ['phone', 'address', 'postcode', 'reference', 'id']


def parseGoal(goal, d, domain):
    # parse goal into dict format
    goal[domain] = {}
    goal[domain] = {'informable': [], 'requestable': [], 'booking': []}
    if d['goal'][domain].has_key('info'):
        if domain == 'train':
            # we consider dialogues only where train had to be booked!
            if d['goal'][domain].has_key('book'):
                goal[domain]['requestable'].append('reference')
            if d['goal'][domain].has_key('reqt'):
                if 'trainID' in d['goal'][domain]['reqt']:
                    goal[domain]['requestable'].append('id')
        else:
            if d['goal'][domain].has_key('reqt'):
                for s in d['goal'][domain]['reqt']:  # addtional requests:
                    if s in ['phone', 'address', 'postcode', 'reference', 'id']:
                        # ones that can be easily delexicalized
                        goal[domain]['requestable'].append(s)
            if d['goal'][domain].has_key('book'):
                goal[domain]['requestable'].append("reference")

        goal[domain]["informable"] = d['goal'][domain]['info']
        if d['goal'][domain].has_key('book'):
            goal[domain]["booking"] = d['goal'][domain]['book']

    return goal


def evaluateModel(dialogues, val_dials, mode='valid'):
    fin1 = file('../data/multi-woz/delex.json')
    delex_dialogues = json.load(fin1)
    successes, matches = 0, 0
    total = 0

    gen_stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0,0], 'taxi': [0, 0, 0],
             'hospital': [0, 0, 0], 'police': [0, 0, 0]}
    sng_gen_stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                     'taxi': [0, 0, 0],
                     'hospital': [0, 0, 0], 'police': [0, 0, 0]}

    for filename, dial in dialogues.items():
        data = delex_dialogues[filename]

        goal, _, _, requestables, _ = evaluateRealDialogue(data, filename)

        success, match, stats = evaluateGeneratedDialogue(dial, goal, data, requestables)

        successes += success
        matches += match
        total += 1

        for domain in gen_stats.keys():
            gen_stats[domain][0] += stats[domain][0]
            gen_stats[domain][1] += stats[domain][1]
            gen_stats[domain][2] += stats[domain][2]

        if 'SNG' in filename:
            for domain in gen_stats.keys():
                sng_gen_stats[domain][0] += stats[domain][0]
                sng_gen_stats[domain][1] += stats[domain][1]
                sng_gen_stats[domain][2] += stats[domain][2]

    # BLUE SCORE
    corpus = []
    model_corpus = []
    bscorer = BLEUScorer()

    for dialogue in dialogues:
        data = val_dials[dialogue]
        model_turns, corpus_turns = [], []
        for idx, turn in enumerate(data['sys']):
            corpus_turns.append([turn])
        for turn in dialogues[dialogue]:
            model_turns.append([turn])

        if len(model_turns) == len(corpus_turns):
            corpus.extend(corpus_turns)
            model_corpus.extend(model_turns)
        else:
            print('wrong length!!!')
            print(model_turns)

    # Print results
    if mode=='valid':
        try: print "Valid BLUES SCORE %.10f" % bscorer.score(model_corpus, corpus)
        except: print('BLUE SCORE ERROR')
        print 'Valid Corpus Matches : %2.2f%%' % (matches / float(total) * 100)
        print 'Valid Corpus Success : %2.2f%%' %  (successes / float(total) * 100)
        print 'Valid Total number of dialogues: %s ' % total
    else:
        try:
            print "Corpus BLUES SCORE %.10f" % bscorer.score(model_corpus, corpus)
        except:
            print('BLUE SCORE ERROR')
        print 'Corpus Matches : %2.2f%%' % (matches / float(total) * 100)
        print 'Corpus Success : %2.2f%%' % (successes / float(total) * 100)
        print 'Total number of dialogues: %s ' % total


def evaluateRealDialogue(dialog, filename):
    domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
    requestables = ['phone', 'address', 'postcode', 'reference', 'id']

    # get the list of domains in the goal
    domains_in_goal = []
    goal = {}
    for domain in domains:
        if dialog['goal'][domain]:
            goal = parseGoal(goal, dialog, domain)
            domains_in_goal.append(domain)

    # compute corpus success
    real_requestables = {}
    provided_requestables = {}
    venue_offered = {}
    for domain in goal.keys():
        provided_requestables[domain] = []
        venue_offered[domain] = []
        real_requestables[domain] = goal[domain]['requestable']

    # iterate each turn
    m_targetutt = [turn['text'] for idx, turn in enumerate(dialog['log']) if idx % 2 == 1]
    for t in range(len(m_targetutt)):
        for domain in domains_in_goal:
            sent_t = m_targetutt[t]
            # for computing match - where there are limited entities
            if domain + '_name' in sent_t or '_id' in sent_t:
                if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                    # HERE YOU CAN PUT YOUR BELIEF STATE ESTIMATION
                    venues = queryResultVenues(domain, dialog['log'][t * 2 + 1])

                    # if venue has changed
                    if len(venue_offered[domain]) == 0 and venues:
                        venue_offered[domain] = random.sample(venues, 1)
                    else:
                        flag = False
                        for ven in venues:
                            if venue_offered[domain][0] == ven:
                                flag = True
                                break
                        if not flag and venues:  # sometimes there are no results so sample won't work
                            #print venues
                            venue_offered[domain] = random.sample(venues, 1)
                else:  # not limited so we can provide one
                    venue_offered[domain] = '[' + domain + '_name]'

            for requestable in requestables:
                # check if reference could be issued
                if requestable == 'reference':
                    if domain + '_reference' in sent_t:
                        if 'restaurant_reference' in sent_t:
                            if dialog['log'][t * 2]['db_pointer'][-5] == 1:  # if pointer was allowing for that?
                                provided_requestables[domain].append('reference')

                        elif 'hotel_reference' in sent_t:
                            if dialog['log'][t * 2]['db_pointer'][-3] == 1:  # if pointer was allowing for that?
                                provided_requestables[domain].append('reference')

                                #return goal, 0, match, real_requestables
                        elif 'train_reference' in sent_t:
                            if dialog['log'][t * 2]['db_pointer'][-1] == 1:  # if pointer was allowing for that?
                                provided_requestables[domain].append('reference')

                        else:
                            provided_requestables[domain].append('reference')
                else:
                    if domain + '_' + requestable in sent_t:
                        provided_requestables[domain].append(requestable)

    # offer was made?
    for domain in domains_in_goal:
        # if name was provided for the user, the match is being done automatically
        if dialog['goal'][domain].has_key('info'):
            if dialog['goal'][domain]['info'].has_key('name'):
                venue_offered[domain] = '[' + domain + '_name]'

        # special domains - entity does not need to be provided
        if domain in ['taxi', 'police', 'hospital']:
            venue_offered[domain] = '[' + domain + '_name]'

        # if id was not requested but train was found we dont want to override it to check if we booked the right train
        if domain == 'train' and (not venue_offered[domain] and 'id' not in goal['train']['requestable']):
            venue_offered[domain] = '[' + domain + '_name]'

    # HARD (0-1) EVAL
    stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0,0], 'taxi': [0, 0, 0],
             'hospital': [0, 0, 0], 'police': [0, 0, 0]}

    match, success = 0, 0
    # MATCH
    for domain in goal.keys():
        match_stat = 0
        if domain in ['restaurant', 'hotel', 'attraction', 'train']:
            goal_venues = queryResultVenues(domain, dialog['goal'][domain]['info'], real_belief=True)
            #print(goal_venues)
            if len(venue_offered[domain]) > 0 and venue_offered[domain][0] in goal_venues:
                match += 1
                match_stat = 1
            elif '_name' in venue_offered[domain]:
                match += 1
                match_stat = 1
        else:
            if '[' + domain + '_name]' in venue_offered[domain]:
                match += 1
                match_stat = 1

        stats[domain][0] = match_stat
        stats[domain][2] = 1

    if match == len(goal.keys()):
        match = 1
    else:
        match = 0

    # SUCCESS
    if match:
        for domain in domains_in_goal:
            domain_success = 0
            success_stat = 0
            if len(real_requestables[domain]) == 0:
                # check that
                success += 1
                success_stat = 1
                stats[domain][1] = success_stat
                continue
            # if values in sentences are super set of requestables
            for request in set(provided_requestables[domain]):
                if request in real_requestables[domain]:
                    domain_success += 1

            if domain_success >= len(real_requestables[domain]):
                success +=1
                success_stat = 1

            stats[domain][1] = success_stat

        # final eval
        if success >= len(real_requestables):
            success = 1
        else:
            success = 0

    return goal, success, match, real_requestables, stats


def evaluateGeneratedDialogue(dialog, goal, realDialogue, real_requestables):
    # for computing corpus success
    requestables = ['phone', 'address', 'postcode', 'reference', 'id']

    # CHECK IF MATCH HAPPENED
    provided_requestables = {}
    venue_offered = {}
    domains_in_goal = []

    for domain in goal.keys():
        venue_offered[domain] = []
        provided_requestables[domain] = []
        domains_in_goal.append(domain)

    for t, sent_t in enumerate(dialog):
        for domain in goal.keys():
            # for computing success
            if '[' + domain + '_name]' in sent_t or '_id' in sent_t:
                if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                    # HERE YOU CAN PUT YOUR BELIEF STATE ESTIMATION
                    venues = queryResultVenues(domain, realDialogue['log'][t*2 + 1])

                    # if venue has changed
                    if len(venue_offered[domain]) == 0 and venues:
                        venue_offered[domain] = random.sample(venues, 1)
                    else:
                        flag = False
                        for ven in venues:
                            if venue_offered[domain][0] == ven:
                                flag = True
                                break
                        if not flag and venues:  # sometimes there are no results so sample won't work
                            # print venues
                            venue_offered[domain] = random.sample(venues, 1)
                else:  # not limited so we can provide one
                    venue_offered[domain] = '[' + domain + '_name]'

            # ATTENTION: assumption here - we didn't provide phone or address twice! etc
            for requestable in requestables:
                if requestable == 'reference':
                    if domain + '_reference' in sent_t:
                        if 'restaurant_reference' in sent_t:
                            if realDialogue['log'][t * 2]['db_pointer'][-5] == 1:  # if pointer was allowing for that?
                                provided_requestables[domain].append('reference')

                        elif 'hotel_reference' in sent_t:
                            if realDialogue['log'][t * 2]['db_pointer'][-3] == 1:  # if pointer was allowing for that?
                                provided_requestables[domain].append('reference')

                        elif 'train_reference' in sent_t:
                            if realDialogue['log'][t * 2]['db_pointer'][-1] == 1:  # if pointer was allowing for that?
                                provided_requestables[domain].append('reference')

                        else:
                            provided_requestables[domain].append('reference')
                else:
                    if domain + '_' + requestable + ']' in sent_t:
                        provided_requestables[domain].append(requestable)

    # if name was given in the task
    for domain in goal.keys():
        # if name was provided for the user, the match is being done automatically
        if realDialogue['goal'][domain].has_key('info'):
            if realDialogue['goal'][domain]['info'].has_key('name'):
                venue_offered[domain] = '[' + domain + '_name]'

        # special domains - entity does not need to be provided
        if domain in ['taxi', 'police', 'hospital']:
            venue_offered[domain] = '[' + domain + '_name]'


        if domain == 'train':
            if not venue_offered[domain]:
                if realDialogue['goal'][domain].has_key('reqt') and 'id' not in realDialogue['goal'][domain]['reqt']:
                    venue_offered[domain] = '[' + domain + '_name]'

    # HARD EVAL
    stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0,0], 'taxi': [0, 0, 0],
             'hospital': [0, 0, 0], 'police': [0, 0, 0]}

    match = 0
    success = 0
    # MATCH
    for domain in goal.keys():
        match_stat = 0
        if domain in ['restaurant', 'hotel', 'attraction', 'train']:
            goal_venues = queryResultVenues(domain, goal[domain]['informable'], real_belief=True)
            if len(venue_offered[domain]) > 0 and venue_offered[domain][0] in goal_venues:
                match += 1
                match_stat = 1
            elif '_name' in venue_offered[domain]:
                match += 1
                match_stat = 1
        else:
            if domain + '_name]' in venue_offered[domain]:
                match += 1
                match_stat = 1

        stats[domain][0] = match_stat
        stats[domain][2] = 1

    if match == len(goal.keys()):
        match = 1
    else:
        match = 0

    # SUCCESS
    if match:
        for domain in domains_in_goal:
            success_stat = 0
            domain_success = 0
            if len(real_requestables[domain]) == 0:
                success += 1
                success_stat = 1
                stats[domain][1] = success_stat
                continue
            # if values in sentences are super set of requestables
            for request in set(provided_requestables[domain]):
                if request in real_requestables[domain]:
                    domain_success += 1

            if domain_success >= len(real_requestables[domain]):
                success += 1
                success_stat = 1

            stats[domain][1] = success_stat

        # final eval
        if success >= len(real_requestables):
            success = 1
        else:
            success = 0

    #rint requests, 'DIFF', requests_real, 'SUCC', success
    return success, match, stats
