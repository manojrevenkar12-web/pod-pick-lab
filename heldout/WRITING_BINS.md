# Writing a held-out bin family

You are writing a test set for a robot that picks one ordered item out of a fabric storage-pod
bin with a suction cup. Write it from this page only. Do not read the code, the configuration,
the other families, the reports or any results. The point of a held-out set is that nobody who
tuned the robot chose it.

## The bin

One bin is an open-fronted fabric box: 0.30 m wide (x), 0.30 m deep (y), 0.22 m high (z). The
robot reaches in from the front. An elastic band runs across the front of the bin at some height
between 0.09 m and 0.15 m (chosen at random per bin, you do not set it).

Items are rectangular. Each bin is filled by dropping items one after another at random positions;
an item that lands on another stays on it. Three kinds of item exist:

- `box`: cardboard, good suction surfaces;
- `bag`: polybags, soft and wrinkled, unreliable suction;
- `book`: dense; stands upright (thin side to the front) or sometimes lies flat.

## What you write

One YAML file, `heldout/heldout.yaml`, with this shape and nothing else:

```yaml
family:
  items: [MIN, MAX]          # number of items dropped into a bin, integers, 3 <= MIN <= MAX <= 14
  mix: {box: P, bag: P, book: P}   # probabilities, any non-negative numbers (normalised)
  box_size: {x: [LO, HI], y: [LO, HI], z: [LO, HI]}    # full edge lengths in metres
  bag_size: {x: [LO, HI], y: [LO, HI], z: [LO, HI]}
  book_size: {x: [LO, HI], y: [LO, HI], z: [LO, HI]}   # x is the thickness of an upright book
  yaw_deg: D                 # items are rotated about the vertical by up to +/- D degrees, 0..30
  front_bias: B              # 0: items spread over the depth; 1: items crowd the front. 0..1
  drop_gap: 0.004            # leave as is
notes: >
  One paragraph: what kind of bin this is meant to represent and why it is a fair but different test.
```

Constraints so the bins are physically possible:
- every size range has 0.02 <= LO <= HI; box and bag x, y <= 0.16; z <= 0.12;
- book x (thickness) <= 0.05, book y <= 0.22, book z <= 0.20;
- choose something a real fulfilment centre could plausibly hold in one bin.

Make it different from a generic mixed bin in at least two ways (for example the item count, the
kind of items, their sizes or how they are spread), but keep it a bin a picking robot should be
expected to handle at least sometimes. Do not try to make it impossible.

When the file exists, it is run once. Its first result is recorded in
`reports/heldout/first_run.json` and is never rewritten, whatever it shows.
