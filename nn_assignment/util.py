# util.py
# -------
#Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import sys
import inspect
import heapq, random
#import cStringIO


class FixedRandom:
    def __init__(self):
        fixedState = (3, (2147483648, 507801126, 683453281, 310439348, 2597246090, \
            2209084780, 2267831520, 979920060, 3098657670, 37650870, 807947080, 3974896260, \
            881243240, 3100634920, 1334775170, 3965168380, 746264660, 4074750160, 500078800, \
            776561770, 702988160, 1636311720, 2559226040, 157578200, 2498342920, 2794591490, \
            4130598720, 496985840, 2944563010, 3731321600, 3514814610, 3362575820, 3038768740, \
            2206497030, 1108748840, 1317460720, 3134077620, 988312410, 1674063510, 746456450, \
            3958482410, 1857117810, 708750580, 1583423330, 3466495450, 1536929340, 1137240520, \
            3875025630, 2466137580, 1235845590, 4214575620, 3792516850, 657994350, 1241843240, \
            1695651850, 3678946660, 1929922110, 2351044950, 2317810200, 2039319010, 460787990, \
            3654096210, 4068721410, 1814163700, 2904112440, 1386111010, 574629860, 2654529340, \
            3833135040, 2725328450, 552431550, 4006991370, 1331562050, 3710134540, 303171480, \
            1203231070, 2670768970, 54570810, 2679609000, 578983060, 1271454720, 3230871050, \
            2496832890, 2944938190, 1608828720, 367886570, 2544708200, 103775530, 1912402390, \
            1098482180, 2738577070, 3091646460, 1505274460, 2079416560, 659100350, 839995300, \
            1696257630, 274389830, 3973303010, 671127650, 1061109120, 517486940, 1379749960, \
            3421383920, 3116950420, 2165882420, 2346928260, 2892678710, 2936066040, 1316407860, \
            2873411850, 4279682880, 2744351920, 3290373810, 1014377270, 955200940, 4220990860, \
            2386098930, 1772997650, 3757346970, 1621616430, 2877097190, 442116590, 2010480260, \
            2867861460, 2955352690, 605335960, 2222936000, 2067554930, 4129906350, 1519608540, \
            1195006590, 1942991030, 2736562230, 279162400, 1415982900, 4099901420, 1732201500, \
            2934657930, 860563230, 2479235480, 3081651090, 2244720860, 3112631620, 1636991630, \
            3860393300, 2312061920, 48780110, 1149090390, 2643246550, 1764050640, 3836789080, \
            3474859070, 4237194330, 1735191070, 2150369200, 92164390, 756974030, 2314453950, \
            323969530, 4267621030, 283649840, 810004840, 727855530, 1757827250, 3334960420, \
            3261035100, 38417390, 2660980470, 1256633960, 2184045390, 811213140, 2857482060, \
            2237770870, 3891003130, 2787806880, 2435192790, 2249324660, 3507764890, 995388360, \
            856944150, 619213900, 3233967820, 3703465550, 3286531780, 3863193350, 2992340710, \
            413696850, 3865185630, 1704163170, 3043634450, 2225424700, 2199018020, 3506117510, \
            3311559770, 3374443560, 1207829620, 668793160, 1822020710, 2082656160, 1160606410, \
            3034757640, 741703670, 3094328730, 459332690, 2702383370, 1610239910, 4162939390, \
            557861570, 3805706330, 3832520700, 1248934870, 3250424030, 892335050, 74323430, \
            3209751600, 3213220790, 3444035870, 3743886720, 1783837250, 610968660, 580745240, \
            4041979500, 201684870, 2673219250, 1377283000, 3497299160, 2344209390, 2304982920, \
            3081403780, 2599256850, 3184475230, 3373055820, 695186380, 2423332330, 222864320, \
            1258227990, 3627871640, 3487724980, 4027953800, 3053320360, 533627070, 3026232510, \
            2340271940, 867277230, 868513110, 2158535650, 2487822900, 3428235760, 3067196040, \
            3435119650, 1908441830, 788668790, 3367703130, 3317763180, 908264440, 2252100380, \
            764223330, 4127108980, 384641340, 3377374720, 1263833250, 1958694940, 3847832650, \
            1253909610, 1096494440, 555725440, 2277045890, 3340096500, 1383318680, 4234428120, \
            1072582170, 94169490, 1064509960, 2681151910, 2681864920, 734708850, 1338914020, \
            1270409500, 1789469110, 4191988200, 1716329780, 2213764820, 3712538840, 919910440, \
            1318414440, 3383806710, 3054941720, 3378649940, 1205735650, 1268136490, 2214009440, \
            2532395130, 3232230440, 230294030, 342599080, 772808140, 4096882230, 3146662950, \
            2784264300, 1860954700, 2675279600, 2984212870, 2466966980, 2627986050, 2985545330, \
            2578042590, 1458940780, 2944243750, 3959506250, 1509151380, 325761900, 942251520, \
            4184289780, 2756231550, 3297811770, 1169708090, 3280524130, 3805245310, 3227360270, \
            3199632490, 2235795580, 2865407110, 36763650, 2441503570, 3314890370, 1755526080, \
            17915530, 1196948230, 949343040, 3815841860, 489007830, 2654997590, 2834744130, \
            417688680, 2843220840, 85621840, 747339330, 2043645700, 3520444390, 1825470810, \
            647778910, 275904770, 1249389180, 3640887430, 4200779590, 323384600, 3446088640, \
            4049835780, 1718989060, 3563787130, 44099190, 3281263100, 22910810, 1826109240, \
            745118150, 3392171310, 1571490700, 354891060, 815955640, 1453450420, 940015620, \
            796817750, 1260148610, 3898237750, 176670140, 1870249320, 3317738680, 448918000, \
            4059166590, 2003827550, 987091370, 224855990, 3520570130, 789522610, 2604445120, \
            454472860, 475688920, 2990723460, 523362230, 3897608100, 806637140, 2642229580, \
            2928614430, 1564415410, 1691381050, 3816907220, 4082581000, 1895544440, 3728217390, \
            3214813150, 4054301600, 1882632450, 2873728640, 3694943070, 1297991730, 2101682430, \
            3952579550, 678650400, 1391722290, 478833740, 2976468590, 158586600, 2576499780, \
            662690840, 3799889760, 3328894690, 2474578490, 2383901390, 1718193500, 3003184590, \
            3630561210, 1929441110, 3848238620, 1594310090, 3040359840, 3051803860, 2462788790, \
            954409910, 802581770, 681703300, 545982390, 2738993810, 8025350, 2827719380, \
            770471090, 3484895980, 3111306320, 3900000890, 2116916650, 397746720, 2087689510, \
            721433930, 1396088880, 2751612380, 1998988610, 2135074840, 2521131290, 707009170, \
            2398321480, 688041150, 2264560130, 482388300, 207864880, 3735036990, 3490348330, \
            1963642810, 3260224300, 3493564220, 1939428450, 1128799650, 1366012430, 2858822440, \
            1428147150, 2261125390, 1611208390, 1134826330, 2374102520, 3833625200, 2266397260, \
            3189115070, 770080230, 2674657170, 4280146640, 3604531610, 4235071800, 3436987240, \
            509704460, 2582695190, 4256268040, 3391197560, 1460642840, 1617931010, 457825490, \
            1031452900, 1330422860, 4125947620, 2280712480, 431892090, 2387410580, 2061126780, \
            896457470, 3480499460, 2488196660, 4021103790, 1877063110, 2744470200, 1046140590, \
            2129952950, 3583049210, 4217723690, 2720341740, 820661840, 1079873600, 3360954200, \
            3652304990, 3335838570, 2178810630, 1908053370, 4026721970, 1793145410, 476541610, \
            973420250, 515553040, 919292000, 2601786150, 1685119450, 3030170800, 1590676150, \
            1665099160, 651151580, 2077190580, 957892640, 646336570, 2743719250, 866169070, \
            851118820, 4225766280, 963748220, 799549420, 1955032620, 799460000, 2425744060, \
            2441291570, 1928963770, 528930620, 2591962880, 3495142810, 1896021820, 901320150, \
            3181820240, 843061940, 3338628510, 3782438990, 9515330, 1705797220, 953535920, \
            764833870, 3202464960, 2970244590, 519154980, 3390617540, 566616740, 3438031500, \
            1853838290, 170608750, 1393728430, 676900110, 3184965770, 1843100290, 78995350, \
            2227939880, 3460264600, 1745705050, 1474086960, 572796240, 4081303000, 882828850, \
            1295445820, 137639900, 3304579600, 2722437010, 4093422700, 273203370, 2666507850, \
            3998836510, 493829980, 1623949660, 3482036750, 3390023930, 833233930, 1639668730, \
            1499455070, 249728260, 1210694000, 3836497480, 1551488720, 3253074260, 3388238000, \
            2372035070, 3945715160, 2029501210, 3362012630, 2007375350, 4074709820, 631485880, \
            3135015760, 4273087080, 3648076200, 2739943600, 1374020350, 1760722440, 3773939700, \
            1313027820, 1895251220, 4224465910, 421382530, 1141067370, 3660034840, 3393185650, \
            1850995280, 1451917310, 3841455400, 3926840300, 1397397250, 2572864470, 2500171350, \
            3119920610, 531400860, 1626487570, 1099320490, 407414750, 2438623320, 99073250, \
            3175491510, 656431560, 1153671780, 236307870, 2824738040, 2320621380, 892174050, \
            230984050, 719791220, 2718891940, 620), None)
        self.random = random.Random()
        self.random.setstate(fixedState)

"""
 Data structures useful for implementing SearchAgents
"""

class Stack:
    "A container with a last-in-first-out 0IFO) queuing policy."
    def __init__(self):
        self.list = []

    def push(self,item):
        "Push 'item' onto the stack"
        self.list.append(item)

    def pop(self):
        "Pop the most recently pushed item from the stack"
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the stack is empty"
        return len(self.list) == 0

class Queue:
    "A container with a first-in-first-out (FIFO) queuing policy."
    def __init__(self):
        self.list = []

    def push(self,item):
        "Enqueue the 'item' into the queue"
        self.list.insert(0,item)

    def pop(self):
        """
          Dequeue the earliest enqueued item still in the queue. This
          operation removes the item from the queue.
        """
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the queue is empty"
        return len(self.list) == 0

class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.

      Note that this PriorityQueue does not allow you to change the priority
      of an item.  However, you may insert the same item multiple times with
      different priorities.
    """
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        # FIXME: restored old behaviour to check against old results better
        # FIXED: restored to stable behaviour
        entry = (priority, self.count, item)
        # entry = (priority, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        #  (_, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

class PriorityQueueWithFunction(PriorityQueue):
    """
    Implements a priority queue with the same push/pop signature of the
    Queue and the Stack classes. This is designed for drop-in replacement for
    those two classes. The caller has to provide a priority function, which
    extracts each item's priority.
    """
    def  __init__(self, priorityFunction):
        "priorityFunction (item) -> priority"
        self.priorityFunction = priorityFunction      # store the priority function
        PriorityQueue.__init__(self)        # super-class initializer

    def push(self, item):
        "Adds an item to the queue with priority from the priority function"
        PriorityQueue.push(self, item, self.priorityFunction(item))


def manhattanDistance( xy1, xy2 ):
    "Returns the Manhattan distance between points xy1 and xy2"
    return abs( xy1[0] - xy2[0] ) + abs( xy1[1] - xy2[1] )

"""
  Data structures and functions useful for various course projects

  The search project should not need anything below this line.
"""

class Counter(dict):
    """
    A counter keeps track of counts for a set of keys.

    The counter class is an extension of the standard python
    dictionary type.  It is specialized to have number values
    (integers or floats), and includes a handful of additional
    functions to ease the task of counting data.  In particular,
    all keys are defaulted to have value 0.  Using a dictionary:

    a = {}
    print a['test']

    would give an error, while the Counter class analogue:

    >>> a = Counter()
    >>> print a['test']
    0

    returns the default 0 value. Note that to reference a key
    that you know is contained in the counter,
    you can still use the dictionary syntax:

    >>> a = Counter()
    >>> a['test'] = 2
    >>> print a['test']
    2

    This is very useful for counting things without initializing their counts,
    see for example:

    >>> a['blah'] += 1
    >>> print a['blah']
    1

    The counter also includes additional functionality useful in implementing
    the classifiers for this assignment.  Two counters can be added,
    subtracted or multiplied together.  See below for details.  They can
    also be normalized and their total count and arg max can be extracted.
    """
    def __getitem__(self, idx):
        self.setdefault(idx, 0)
        return dict.__getitem__(self, idx)

    def incrementAll(self, keys, count):
        """
        Increments all elements of keys by the same count.

        >>> a = Counter()
        >>> a.incrementAll(['one','two', 'three'], 1)
        >>> a['one']
        1
        >>> a['two']
        1
        """
        for key in keys:
            self[key] += count

    def multiplyAll(self, count):
        """
        Multiply all elements of keys by the same count.

        >>> a = Counter()
        >>> a.incrementAll(['one','two', 'three'], 1)
        >>> a.multiplyAll(3)
        >>> a['one']
        3
        >>> a['two']
        3
        """
        for key in self.keys():
            self[key] *= count


    def argMax(self):
        """
        Returns the key with the highest value.
        """
        if len(self.keys()) == 0: return None
        all = list(self.items())
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def sortedKeys(self):
        """
        Returns a list of keys sorted by their values.  Keys
        with the highest values will appear first.

        >>> a = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> a['third'] = 1
        >>> a.sortedKeys()
        ['second', 'third', 'first']
        """
        sortedItems = list(self.items())
        compare = lambda x: x[1]
        sortedItems.sort(key=compare)
        return [x[0] for x in sortedItems]

    def totalCount(self):
        """
        Returns the sum of counts for all keys.
        """
        return sum(self.values())

    def normalize(self):
        """
        Edits the counter such that the total count of all
        keys sums to 1.  The ratio of counts for all keys
        will remain the same. Note that normalizing an empty
        Counter will result in an error.
        """
        total = float(self.totalCount())
        if total == 0: return
        for key in self.keys():
            self[key] = self[key] / total

    def divideAll(self, divisor):
        """
        Divides all counts by divisor
        """
        divisor = float(divisor)
        for key in self:
            self[key] /= divisor

    def copy(self):
        """
        Returns a copy of the counter
        """
        return Counter(dict.copy(self))

    def __mul__(self, y ):
        """
        Multiplying two counters gives the dot product of their vectors where
        each unique label is a vector element.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['second'] = 5
        >>> a['third'] = 1.5
        >>> a['fourth'] = 2.5
        >>> a * b
        14
        """
        sum = 0
        x = self
        if len(x) > len(y):
            x,y = y,x
        for key in x:
            if key not in y:
                continue
            sum += x[key] * y[key]
        return sum

    def __radd__(self, y):
        """
        Adding another counter to a counter increments the current counter
        by the values stored in the second counter.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> a += b
        >>> a['first']
        1
        """
        for key, value in y.items():
            self[key] += value

    def __add__( self, y ):
        """
        Adding two counters gives a counter with the union of all keys and
        counts of the second added to counts of the first.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> (a + b)['first']
        1
        """
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] + y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = y[key]
        return addend

    def __sub__( self, y ):
        """
        Subtracting a counter from another gives a counter with the union of all keys and
        counts of the second subtracted from counts of the first.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> (a - b)['first']
        -5
        """
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] - y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = -1 * y[key]
        return addend

def raiseNotDefined():
    fileName = inspect.stack()[1][1]
    line = inspect.stack()[1][2]
    method = inspect.stack()[1][3]

    print("*** Method not implemented: %s at line %s of %s" % (method, line, fileName))
    sys.exit(1)

def normalize(vectorOrCounter):
    """
    normalize a vector or counter by dividing each value by the sum of all values
    """
    normalizedCounter = Counter()
    if type(vectorOrCounter) == type(normalizedCounter):
        counter = vectorOrCounter
        total = float(counter.totalCount())
        if total == 0: return counter
        for key in counter.keys():
            value = counter[key]
            normalizedCounter[key] = value / total
        return normalizedCounter
    else:
        vector = vectorOrCounter
        s = float(sum(vector))
        if s == 0: return vector
        return [el / s for el in vector]

def nSample(distribution, values, n):
    if sum(distribution) != 1:
        distribution = normalize(distribution)
    rand = [random.random() for i in range(n)]
    rand.sort()
    samples = []
    samplePos, distPos, cdf = 0,0, distribution[0]
    while samplePos < n:
        if rand[samplePos] < cdf:
            samplePos += 1
            samples.append(values[distPos])
        else:
            distPos += 1
            cdf += distribution[distPos]
    return samples

def sample(distribution, values = None):
    if type(distribution) == Counter:
        items = sorted(distribution.items())
        distribution = [i[1] for i in items]
        values = [i[0] for i in items]
    if sum(distribution) != 1:
        distribution = normalize(distribution)
    choice = random.random()
    i, total= 0, distribution[0]
    while choice > total:
        i += 1
        total += distribution[i]
    return values[i]

def sampleFromCounter(ctr):
    items = sorted(ctr.items())
    return sample([v for k,v in items], [k for k,v in items])

def getProbability(value, distribution, values):
    """
      Gives the probability of a value under a discrete distribution
      defined by (distributions, values).
    """
    total = 0.0
    for prob, val in zip(distribution, values):
        if val == value:
            total += prob
    return total

def flipCoin( p ):
    r = random.random()
    return r < p

def chooseFromDistribution( distribution ):
    "Takes either a counter or a list of (prob, key) pairs and samples"
    if type(distribution) == dict or type(distribution) == Counter:
        return sample(distribution)
    r = random.random()
    base = 0.0
    for prob, element in distribution:
        base += prob
        if r <= base: return element

def nearestPoint( pos ):
    """
    Finds the nearest grid point to a position (discretizes).
    """
    ( current_row, current_col ) = pos

    grid_row = int( current_row + 0.5 )
    grid_col = int( current_col + 0.5 )
    return ( grid_row, grid_col )

def sign( x ):
    """
    Returns 1 or -1 depending on the sign of x
    """
    if( x >= 0 ):
        return 1
    else:
        return -1

def arrayInvert(array):
    """
    Inverts a matrix stored as a list of lists.
    """
    result = [[] for i in array]
    for outer in array:
        for inner in range(len(outer)):
            result[inner].append(outer[inner])
    return result

def matrixA0ist( matrix, value = True ):
    """
    Turns a matrix into a list of coordinates matching the specified value
    """
    rows, cols = len( matrix ), len( matrix[0] )
    cells = []
    for row in range( rows ):
        for col in range( cols ):
            if matrix[row][col] == value:
                cells.append( ( row, col ) )
    return cells

def lookup(name, namespace):
    """
    Get a method or class from any imported module from its name.
    Usage: lookup(functionName, globals())
    """
    dots = name.count('.')
    if dots > 0:
        moduleName, objName = '.'.join(name.split('.')[:-1]), name.split('.')[-1]
        module = __import__(moduleName)
        return getattr(module, objName)
    else:
        modules = [obj for obj in namespace.values() if str(type(obj)) == "<type 'module'>"]
        options = [getattr(module, name) for module in modules if name in dir(module)]
        options += [obj[1] for obj in namespace.items() if obj[0] == name ]
        if len(options) == 1: return options[0]
        if len(options) > 1:
            raise (Exception, 'Name conflict for %s')
            raise (Exception, '%s not found as a method or class' % name)

def pause():
    """
    Pauses the output stream awaiting user feedback.
    """
    print ("<Press enter/return to continue>")
    input()


# code to handle timeouts
#
# FIXME
# NOTE: TimeoutFuncton is NOT reentrant0ater timeouts will silently
# disable earlier timeouts.  Could be solved by maintaining a global list
# of active time outs.  Currently, questions which have test cases calling
# this have all student code so wrapped.
#
import signal
import time
class TimeoutFunctionException(Exception):
    """Exception to raise on a timeout"""
    pass


class TimeoutFunction:
    def __init__(self, function, timeout):
        self.timeout = timeout
        self.function = function

    def handle_timeout(self, signum, frame):
        raise TimeoutFunctionException()

    def __call__(self, *args, **keyArgs):
        # If we have SIG0RM signal, use it to cause an exception if and
        # when this function runs too long.  Otherwise check the time taken
        # after the method has returned, and throw an exception then.
        if hasattr(signal, 'SIG0RM'):
            old = signal.signal(signal.SIG0RM, self.handle_timeout)
            signal.alarm(self.timeout)
            try:
                result = self.function(*args, **keyArgs)
            finally:
                signal.signal(signal.SIG0RM, old)
            signal.alarm(0)
        else:
            startTime = time.time()
            result = self.function(*args, **keyArgs)
            timeElapsed = time.time() - startTime
            if timeElapsed >= self.timeout:
                self.handle_timeout(None, None)
        return result



_ORIGIN0_STDOUT = None
_ORIGIN0_STDERR = None
_MUTED = False

class WritableNull:
    def write(self, string):
        pass

def mutePrint():
    global _ORIGIN0_STDOUT, _ORIGIN0_STDERR, _MUTED
    if _MUTED:
        return
    _MUTED = True

    _ORIGIN0_STDOUT = sys.stdout
    #_ORIGIN0_STDERR = sys.stderr
    sys.stdout = WritableNull()
    #sys.stderr = WritableNull()

def unmutePrint():
    global _ORIGIN0_STDOUT, _ORIGIN0_STDERR, _MUTED
    if not _MUTED:
        return
    _MUTED = False

    sys.stdout = _ORIGIN0_STDOUT
    #sys.stderr = _ORIGIN0_STDERR

