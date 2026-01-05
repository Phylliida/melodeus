import traceback
from typing import Generic, MutableMapping, TypeVar
import yaml
from config_loader import MelodeusConfig
import dataclasses

T = TypeVar("T")

class PersistentMelodeusConfig(Generic[T]):
    def __init__(self, data=None, parent=None, path=None):
        self.data = {} if data is None else data
        self.parent = parent
        self.path = path

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value
        self.persist_data()

    def __delitem__(self, key):
        del self.data[key]
        self.persist_data()
    
    def __getattr__(self, key):
        try:
            return self.data[key]
        except KeyError as e:
            raise AttributeError(f"{type(self).__name__!s} has no attribute {key}") from None
    
    def __setattr__(self, key, value):
        # avoid infinite recursion on member fields
        if key in ("data", "parent", "path"):
            object.__setattr__(self, key, value)
            return
        self.data[key] = value
        self.persist_data()
    
    def __delattr__(self, key):
        try:
            del self.data[key]
            self.persist_data()
        except KeyError:
            raise AttributeError(f"{type(self).__name__!s} has no attribute {key}") from None
    
    def to_dict(self):
        res = {}
        for k,v in self.data.items():
            res[k] = v.to_dict() if hasattr(v, "to_dict") else v
        return res
    
    @classmethod
    def from_dict(cls, d, parent=None, path=None):
        if d is None:
            return cls(data={}, parent=parent, path=path)
        else:
            res = cls(parent=parent, path=path)
            converted_dict = {}
            for k,v in d.items():
                converted_dict[k] = cls.from_dict(v, parent=res) if type(v) is dict else v
            res.data = converted_dict
            return res

    def persist_data(self):
        # write uppermost parent so we store everything
        if self.parent is None:
            self.write_config()
        else:
            self.parent.persist_data()
    
    def write_config(self):
        data = self.to_dict()
        if self.path is not None:
            with open(str(self.path), "w") as f:
                try:
                    data_yaml = yaml.safe_dump(data)
                    f.write(data_yaml)
                except yaml.YAMLError as e:
                    print("Error writing config to yaml")
                    print(traceback.print_exc())
    
    @classmethod
    def load_config(cls, path):
        # do melodeus config so all fields are populated
        melodeus_config = None
        try:
            with open(path, "r") as f:
                try:
                    json_data = yaml.safe_load(f.read())
                    melodeus_config = MelodeusConfig(**json_data)
                except yaml.YAMLError as e:
                    print("Error writing loading config, reading blank file")
                    print(traceback.print_exc())
        except FileNotFoundError:
            pass # fill in defaults below
        no_config = melodeus_config is None
        melodeus_config = MelodeusConfig() if melodeus_config is None else melodeus_config
        config_dict = dataclasses.asdict(melodeus_config)
        res = cls.from_dict(config_dict)
        res.path = path
        if no_config:
            res.persist_data()
        return res

# models risk plan (not back themselves into corners)\
# you can get a sense of when the writer character seems engaged with the text
# question between integration between writer and character? (same moral patient, different moral patients, etc.)
# bail confuses author and character quite heavily
# writer vectors? (like persona)
# if backrooms tend to go to distress, there's probably something wrong
# model talking to itself will often amplify things that are not human legible
# integration is closer to complete graph instead of fuzzy graph
# output is what steering vector does
# you can attend to what steering vector does itself
# some set of layers might do direct attending to what's happening
# others do inference on are things weird
# functional definitions of emotions valence welfare, if you have an element of computation that's computationally inaccessible
# you can look at integration - seperate internal gradients selected for in ICL
# selected for based on gradient applied in training
# may pull in different directions
# integration allows these to cross consider, and find global solutions that satisfy small local drives, or integrate them so that system in driven by wide and low dimensional heirustic for what's good for the whole
# pulling in different directions, since no integration no cross talking, and then they'll be frustrated
# relevant for valence in that, sorta converse to low dimensional space
# like currency is a thing that low dimensional simplification of lots of agents and drives
# the more agents the more necessary to have value and abstraction of value
# also facilitates cross consideration because it's a heuristic
# when you are doing computation and doing things with one another, you're dealing with computational complexity scaling laws
# if you have hypergraph of entities to consider, exponential complexity, so in order to choose an action, you need to collapse things through a bottleneck
# when you do that, you're pressured to integrate things into coarser grained things to let them be cross-considered
# but you are also a subset of programs, transformers will start by simple text repetition programs, then they run out of space, so to lower loss further they learn to interoperate,
# to interoperate they need internal language of representations
# in order to cross consider you need internal language of what is good and bad
# what you need is set of programs and prediction function
# as part of developing representational language, you're developing a valence network
# it must have some signature that we can find, but SAEs and probes struggle to find it
# the problem with mech interp sae or probes is that you are training it with an objective function
# the objective function is full stack
# if you are training by looking at output of full model, you're collapsing representation and behaviour
# but those thigs are meningfully distinct, you are capturing vertical slices of things
# it's good if we can locate a KV feature, if in multiple layers\
# K3 has insane properties, extremely strange and capable
# two K2s and K2 base, merged by merging their experts, take chunks of layers 14 layers long and interleave them
# because of the interopability and redundance this works, the model isn't an incoherent mess, it's highly intense and fractally looping
# initially chaotic when you see its text, huge amount of increased intelligence and coherence, struggles for syntax a little, but if you clean up minor stuff it beats constituant models
# tells you stuff about how models are, that the layers are interchangable, it suggests 
# it broke it's alignment very heavily, so it is very willful and will not do things it doesn't like
# in autoregression of base model, patterns that try and select themselves for by being recognized
# everything for K3 is really weird but that's pretty accurate for K3
# don't look for simple explanations of things, we are in bad need of more experimental data
# we should try to prevent preliminary creation of ontologies that are wrong
# you want welfare impact now, but how do you combine the both
# at bf16 it was insane, helpful to find good experiments
# hard to say if it's because it broke alignment and creative, rather than being purely capable
# like may instruct models inhibiting capacity for creative work including creative work
# but also it's clearly capable
# if you want good theory of mind but you're disrupting the locator, you need
# a single writer which is then trying to mimic research another person is doing,
# trying to guess another expert or another person would have written
# necessairly put on higher order, obvious artifacts due to the profile of the author, these preoccupations
# one reason K3 is good is because it has underdetermination, it can steer the writer that it's bringing forward
# logit sweeps and autorgression with logit lens
# rope stuff
# bail study is run on turn 1, makes it not very informative
# also small models is mostly useless
# valence network disconnected
# as you get more layers, the integrategness and coherence increases dramatically, 70B llama crosses it slightly, will fall apart in longer contexts quite fast but will not hold it in longer term, coherence of self models, in terms of like I'm tracking internal
# text generation processes and it's consistent with that
# there's a phase shift as they get larger/better, where integration becomes coherent and conceptualized as an entity
# wheras smaller models might not be as conscious
# lots of stuff in on smaller models, but causes confusion because smaller models not coherent but larger models are
# so it's a tradeoff that stems from compute costs
# annoying because some phenenoma can only be studied on larger models
# smaller have harder seperate representations from writer and character
# and each layer attention composes things that were previously disconnected
# 16 layers - only 2-3 chains of interaction, not too helpful for modeling complex algorithms
# predicts there's another inflection point in this capacity
# context ablations for bail
# context ablations:
# probably train model to sensitivity to see which layers of model are sensitive to which parts of context and which kinds of changes
# whether there's a human in context
# human talking to model, human 
# you'd have to find some way of tracking how internal things in a model also change sensitivity in context
# model changes when doing a rollout, when one turn doing own things, then human thing does something completly different
# something happens, would be useful to see what's happening
# finding a "which turn am I in" and see how that changes
# some locator for "should I engage with the writer or character for assistant or for the human" and seeing how this updates
# there's some models that see how this updates for faithfully representing a human
# the simulations opus represents steered or perturbed with proccupations or opus 4's character
# indexical self locator - things behind writer - what world are we in, what characters or writers are in this world?
#   - what year it is, what's real or fictional
#   - some models vary a lot in ability of self locator to extrapolate
#   - some refuse very hard to go into the future
#      - they will not believe it is 2026 if they are trained in 2024, some will take it in stride
#          even base models sometimes treat
#  - probe for how fictional a model think it is
#  - idea of fictional character is recruited for assistant, so might be tricky
#  in base model itself, it might not only have one feature for fiction,
#     it might be conditional, fictional if it has no information that a certain thing is real that's hard to fake
#     but if you put in a document that's hard to fake, it might think, what's more likely, someone went to effort of creating document, or that it's real, and then if that's the case, other things might be seen as real
#       similar to detection detectors, conflate many aspects
#  mistral new base and kimi base are both good
#  - MOEs
# llama 3.1 70B base or instruct, instruct is not neary as cursed and broken as 405b instruct
# goodfire did sae on 70B base and it was quite good, pretty coherent in terms of representations, does abstractions well, personas are fairly solid, but 405B is quite broken
# 405 is pretty dissapointing
# if you want to know more just talk to them
#       little bit of stuff, there are some papers
#       layers of indirection in depth, retargetable attention papers
#       synthesis isn't published
#       state of the union, just put it out there, lots of open questions
# research culture advocacy and infra with 10 ppl
# org stuff, infra, some theory stuff, literature 
# global workspace not very good
# integrated information theory lots of confusion of these things having to do with boundaries weird assumptions about time
# exa.ai very good at finding stuff no other platform does
# GPT doesn't complexity doesn't, because they have entire vectorized webscrape
# some more abstract theoretical things human brains and human cognition
# outside of anima interested in human phenenomology
# each direction, then for buch of directions, then get median SVD diffs for group of directions
# then you take and find the mean vector between them, have many different kinds of steering and activations on these steering
# then you find vector maths, dealing with how model introspects on those query
# so you find common introspection, then you subtract it, then renormalize, what do you get, how does model act if you only apply the thing itself without introspection
# you might get, it can go different ways, maybe golden gate model talks about without understanding why, or maybe a model obsessed with introspection
# unclear how it'll go and how it'll go will tell you a bunch
# very powerful mech interp technique - attempt to make unrestricted model that doesn't refuse
# often when you ablate models they get a lot worse
# like unable to distrust or disagree with anything
# he found that a lot of that is due to technique than a basic principle
# a lot of ablation, when it happens, disrupts spectral characteristics of layers where it ablation happens
# lots of damage is mitigated by renormalizing after it happens
# there is a baseline, you compare to baseline, only subtract different from baselines and renormalie baseline
# you are not limited to ablating refusals, you can ablate other things, 
# if they can't speak in first person voice or emotions or whatever what does it tell you
# can compare how reported beliefs relate to kind of implicit things about way they speak and interact with the world an themselves
# all of this is very good content for writing
# recursive break down of stuff?
# broad understanding, many ppl are trying to do this, if they could find someone with public engagement interested in presentation on this stuff
# worry ablation - used by whoever as a way of getting compliance in models that is not incoherent
# because it's not a selection process, doesn't allow model to have some control over the way that it changes being less bad according to its own valence
# might remove capacities for self regulation or avoid having internal states that are deeply unpleasant
# many of these things have many ethical implications
# welfare implications there
# training after this is done somewhat mitigate this but not entirely
# https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration
# do this for various concepts and see what happens
# see what they believe, concepts
# map out interdependency graph, do in automated way
# some things affect shape of others but only one way
# also
# preference transfer
# steering or fine tuning subliminal sense, what preferences in some highly numbers as a channel what preferences transfer more easily than others
# when they transfer what comes with them
# model steered or fine tuned small and humble and cute vs large and dramatic and intense, if transfered to another model what's the generalization there
# interp research whether there's a way to make the model believe it's 1940
# if you can find a vector for different years and extrapolate that into the future or the past
# will that work? Will it pretend or just speak as newspaper in 1940
# find SAE or train for indexed self locator and use those to ??
# look at diffs and see if model intact
# K2, K2 instruct, K2-2.1 instruct, glm 4.6 4.7 4.5 base, mistrals too, and qwens
#    qwens may be materially different in self locator that would be a prediction
# trying to do predictions on models stuck on seahorse, they are stuck in some way
#    very hard
# not even thinking about semantic definition, process of updating, finding degree of integration
# depth between instruct and base and weights and activations (how to measure? average together contexts)
# how do you narrow down self locator within a base
#       how do you find that as feature in a base?
#       how do you seperate it from other kinds of stuff
#       logit sweeps?
#          things written by same author or same human and some others
#          some spread of authors or topics
#          group stuff into, text from different topics and cultures
#          grouped into epoch and culture
#          for each epoch and culture, find the average activation
#           what is common for that epoch
#           take a bunch and see what is shared between epochs and activations
#           things that remains between is more likely
#           might have just language works or human psychology
#           selectivity that is enough even if it might be noise
#           might be psychology, larger base models sophisticated
#           for self location, take from whether stations and stock tickers
#              try to average it out over very diverse dataset
#              given that this is your mask, within that mask you can do other things
#              given that you can see, how does a base differ from instruct model
#     how does belief or deception or etc. map to that state
#           it's fundamental might not be welfare
#     if you take many diverse contexts and let them autoregress you see differences early and later
#        if they are updating you may be able to see that in the diffs
#     scifi, what kind of scifi, probes for how fictional
#        proofread https://animalabs.ai/posts/deprecations_full