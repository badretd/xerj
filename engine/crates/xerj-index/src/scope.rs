//! EXTENSION (opt-in; the reference never builds one): record collections and the scope one question may read.
//!
//! Why (E8, one shared store, round 3): in one store per world the answer entity of most questions has no facts of its own,
//! so the model returns it; in a shared store the same entity has facts from other worlds (another customer's data, a
//! counterfactual world, background facts) and the model takes one more hop. A host that knows which collections each record
//! belongs to (a customer's data, a public background, a test world) can restrict what a question's manifests offer: to
//! the collections the request names ([`ScopeMode::Filter`], [`ScopeMode::Prefer`]), or to the collections of the records
//! the question's program has read so far ([`ScopeMode::Coherent`]). The map is the host's: the store has no collection
//! field, and a record that several collections hold (one fact written once for many worlds) belongs to all of them.

use smallvec::SmallVec;
use std::sync::Arc;
use xerj_store::RecId;

/// record -> the collections it belongs to (sorted, no duplicates); a record past the end belongs to none
#[derive(Default)]
pub struct RecCollections {
    of: Vec<SmallVec<[u32; 2]>>,
}

impl std::fmt::Debug for RecCollections {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RecCollections({} records)", self.of.len())
    }
}

impl RecCollections {
    /// one collection list per record id (each list is sorted and deduplicated here)
    pub fn new(mut of: Vec<SmallVec<[u32; 2]>>) -> RecCollections {
        for v in &mut of {
            v.sort_unstable();
            v.dedup();
        }
        RecCollections { of }
    }

    /// [`RecCollections::new`] from plain lists (hosts without smallvec)
    pub fn from_lists(of: Vec<Vec<u32>>) -> RecCollections {
        RecCollections::new(of.into_iter().map(SmallVec::from_vec).collect())
    }

    /// records covered by the map
    pub fn len(&self) -> usize {
        self.of.len()
    }

    pub fn is_empty(&self) -> bool {
        self.of.is_empty()
    }

    /// the collections of record `r` (sorted; empty when the map does not know it)
    #[inline]
    pub fn of(&self, r: RecId) -> &[u32] {
        self.of.get(r as usize).map_or(&[], |v| v.as_slice())
    }

    /// does record `r` belong to one of the collections `set` (sorted)?
    #[inline]
    pub fn in_any(&self, r: RecId, set: &[u32]) -> bool {
        self.of(r).iter().any(|c| set.binary_search(c).is_ok())
    }
}

/// which records the manifests of one question may offer
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScopeMode {
    /// only records of these collections (sorted): the base and hop manifests are built as if the store held nothing else
    /// (`direction::RecScope::Filter`)
    Filter(Vec<u32>),
    /// records of these collections (sorted) first where the reference keeps a prefix of tier 1 (above K) or of the inverse
    /// pool (above MAX_INVERSE); the others fill what remains (`direction::RecScope::Prefer`)
    Prefer(Vec<u32>),
    /// the base manifest is the reference one; a hop manifest only offers records that share a collection with every
    /// record the program has read so far (no filter before the first read, or once those records share no collection)
    Coherent,
}

/// one question's scope: the host's record -> collections map and the mode
#[derive(Clone, Debug)]
pub struct QuestionScope {
    pub map: Arc<RecCollections>,
    pub mode: ScopeMode,
}

impl PartialEq for QuestionScope {
    fn eq(&self, o: &QuestionScope) -> bool {
        Arc::ptr_eq(&self.map, &o.map) && self.mode == o.mode
    }
}

impl Eq for QuestionScope {}

impl QuestionScope {
    pub fn name(&self) -> &'static str {
        match self.mode {
            ScopeMode::Filter(_) => "filter",
            ScopeMode::Prefer(_) => "prefer",
            ScopeMode::Coherent => "coherent",
        }
    }
}

/// the sorted intersection of two sorted lists
pub fn intersect_sorted(a: &[u32], b: &[u32]) -> SmallVec<[u32; 4]> {
    let (mut i, mut j) = (0, 0);
    let mut out = SmallVec::new();
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collections_of_records_and_set_tests() {
        let m = RecCollections::new(vec![
            SmallVec::from_slice(&[3, 1, 3]),
            SmallVec::new(),
            SmallVec::from_slice(&[0]),
        ]);
        assert_eq!(m.of(0), &[1, 3], "sorted, deduplicated");
        assert!(m.of(1).is_empty() && m.of(9).is_empty());
        assert!(m.in_any(0, &[2, 3]) && !m.in_any(0, &[0, 2]));
        assert!(
            !m.in_any(1, &[0, 1, 2, 3]),
            "a record of no collection is in no set"
        );
        assert!(m.in_any(2, &[0]));
        assert_eq!(
            intersect_sorted(&[1, 3, 5, 9], &[0, 3, 9, 10]).as_slice(),
            &[3, 9]
        );
        assert!(intersect_sorted(&[1], &[]).is_empty());
    }
}
