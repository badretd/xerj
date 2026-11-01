//! memory-world-v1 JSON (memory_world.py, mquake / 2Wiki / FactConsolidation builders) and the exact ingest order of
//! xr_codecs.world_store: base documents, base facts (= the "before" revision), then the edit stream (add_document,
//! correct = add_fact with supersedes, withdraw_document, add_conflict) (= the "after" revision).
//!
//! Parsing is zero-copy where the JSON has no escapes (`Cow::Borrowed` into the input line). Extra fields of the writers
//! (relation_id, serial, key_version, rendering, ...) are ignored, as reference_core.Fact ignores them; subject_id /
//! value_id are kept (stack-v2 id resolution).

use crate::error::ContractError;
use crate::store::{
    AddedFact, DocIx, FactIn, RecId, Rev, Store, ValueIn, DEFAULT_SCOPE, ORACLE_WRITER,
};
use serde::Deserialize;
use smallvec::SmallVec;
use std::borrow::Cow;

#[derive(Deserialize, Debug, Clone)]
pub struct DocJson<'a> {
    #[serde(borrow)]
    pub doc_id: Cow<'a, str>,
    #[serde(borrow)]
    pub version: Cow<'a, str>,
    #[serde(borrow)]
    pub text: Cow<'a, str>,
}

/// a JSON scalar that may be a string, a bool or something else
#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum JsonScalar<'a> {
    Bool(bool),
    Int(i64),
    #[serde(borrow)]
    Str(Cow<'a, str>),
    Other(serde_json::Value),
}

impl<'a> JsonScalar<'a> {
    fn as_value_in(&self) -> ValueIn<'_> {
        match self {
            JsonScalar::Str(s) => ValueIn::Str(s),
            JsonScalar::Bool(b) => ValueIn::Bool(*b),
            _ => ValueIn::Other,
        }
    }
    /// `str(x)` of an id field
    fn id_string(&self) -> Option<Cow<'_, str>> {
        match self {
            JsonScalar::Str(s) => Some(Cow::Borrowed(s)),
            JsonScalar::Int(i) => Some(Cow::Owned(i.to_string())),
            JsonScalar::Bool(b) => Some(Cow::Borrowed(if *b { "True" } else { "False" })),
            JsonScalar::Other(serde_json::Value::Null) => None,
            JsonScalar::Other(v) => Some(Cow::Owned(v.to_string())),
        }
    }
}

#[derive(Deserialize, Debug, Clone)]
pub struct FactJson<'a> {
    #[serde(borrow)]
    pub fact_id: Cow<'a, str>,
    #[serde(borrow)]
    pub subject: Cow<'a, str>,
    #[serde(borrow)]
    pub relation: Cow<'a, str>,
    #[serde(borrow)]
    pub kind: Cow<'a, str>,
    #[serde(borrow)]
    pub value: JsonScalar<'a>,
    #[serde(borrow, default)]
    pub unit: Option<Cow<'a, str>>,
    #[serde(borrow)]
    pub doc_id: Cow<'a, str>,
    #[serde(borrow)]
    pub version: Cow<'a, str>,
    pub start: i64,
    pub end: i64,
    #[serde(borrow)]
    pub quote: Cow<'a, str>,
    #[serde(borrow, default)]
    pub scope: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    pub subject_id: Option<JsonScalar<'a>>,
    #[serde(borrow, default)]
    pub value_id: Option<JsonScalar<'a>>,
    #[serde(borrow, default)]
    pub valid_from: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    pub valid_to: Option<Cow<'a, str>>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct EditJson<'a> {
    #[serde(borrow)]
    pub op: Cow<'a, str>,
    #[serde(borrow, default)]
    pub doc: Option<DocJson<'a>>,
    #[serde(borrow, default)]
    pub facts: Option<Vec<FactJson<'a>>>,
    #[serde(borrow, default)]
    pub fact: Option<FactJson<'a>>,
    #[serde(borrow, default)]
    pub supersedes: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    pub doc_id: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    pub version: Option<Cow<'a, str>>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct ItemJson<'a> {
    #[serde(borrow)]
    pub item_id: Cow<'a, str>,
    #[serde(borrow)]
    pub question: Cow<'a, str>,
    #[serde(borrow, default)]
    pub phase: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    pub case_type: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    pub form: Option<Cow<'a, str>>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct World<'a> {
    #[serde(borrow, default)]
    pub group_id: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    pub world_id: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    pub base_documents: Vec<DocJson<'a>>,
    #[serde(borrow, default)]
    pub base_facts: Vec<FactJson<'a>>,
    #[serde(borrow, default)]
    pub edits: Vec<EditJson<'a>>,
    #[serde(borrow, default)]
    pub items: Vec<ItemJson<'a>>,
}

impl<'a> World<'a> {
    pub fn parse(line: &'a str) -> Result<World<'a>, serde_json::Error> {
        serde_json::from_str(line)
    }

    /// every fact record of the world (base facts and the facts of add_document / correct / add_conflict edits)
    pub fn all_facts(&self) -> impl Iterator<Item = &FactJson<'a>> {
        self.base_facts.iter().chain(
            self.edits
                .iter()
                .flat_map(|e| e.facts.iter().flatten().chain(e.fact.iter())),
        )
    }
}

impl<'a> FactJson<'a> {
    /// the write record of this JSON fact (xr_codecs._fact: unit default "", scope default "default", oracle writer)
    pub fn to_fact_in(&self) -> FactInOwnedIds<'_> {
        FactInOwnedIds {
            sid: self.subject_id.as_ref().and_then(|x| x.id_string()),
            vid: self.value_id.as_ref().and_then(|x| x.id_string()),
            f: self,
        }
    }
}

/// FactIn with id strings that may need owning (numeric JSON ids)
pub struct FactInOwnedIds<'b> {
    sid: Option<Cow<'b, str>>,
    vid: Option<Cow<'b, str>>,
    f: &'b FactJson<'b>,
}

impl<'b> FactInOwnedIds<'b> {
    pub fn get(&self) -> FactIn<'_> {
        let f = self.f;
        FactIn {
            fact_id: &f.fact_id,
            subject: &f.subject,
            relation: &f.relation,
            kind: &f.kind,
            value: f.value.as_value_in(),
            unit: f.unit.as_deref().unwrap_or(""),
            doc_id: &f.doc_id,
            version: &f.version,
            start: f.start,
            end: f.end,
            quote: &f.quote,
            scope: f.scope.as_deref().unwrap_or(DEFAULT_SCOPE),
            subject_id: self.sid.as_deref(),
            value_id: self.vid.as_deref(),
            valid_from: f.valid_from.as_deref(),
            valid_to: f.valid_to.as_deref(),
            writer_version: ORACLE_WRITER,
        }
    }
}

/// Anything that accepts the store write API (the bare `Store`, or an indexed memory that also maintains its indexes).
pub trait WriteSink {
    fn sink_add_document(
        &mut self,
        doc_id: &str,
        version: &str,
        text: &str,
    ) -> Result<DocIx, ContractError>;
    fn sink_add_fact(
        &mut self,
        f: &FactIn<'_>,
        supersedes: &[&str],
    ) -> Result<AddedFact, ContractError>;
    fn sink_withdraw_fact(&mut self, fact_id: &str) -> Result<(RecId, bool), ContractError>;
    fn sink_withdraw_document(
        &mut self,
        doc_id: &str,
        version: &str,
    ) -> Result<(DocIx, SmallVec<[RecId; 8]>), ContractError>;
    fn sink_revision(&self) -> Rev;
}

impl WriteSink for Store {
    fn sink_add_document(
        &mut self,
        doc_id: &str,
        version: &str,
        text: &str,
    ) -> Result<DocIx, ContractError> {
        self.add_document(doc_id, version, text)
    }
    fn sink_add_fact(
        &mut self,
        f: &FactIn<'_>,
        supersedes: &[&str],
    ) -> Result<AddedFact, ContractError> {
        self.add_fact(f, supersedes)
    }
    fn sink_withdraw_fact(&mut self, fact_id: &str) -> Result<(RecId, bool), ContractError> {
        self.withdraw_fact(fact_id)
    }
    fn sink_withdraw_document(
        &mut self,
        doc_id: &str,
        version: &str,
    ) -> Result<(DocIx, SmallVec<[RecId; 8]>), ContractError> {
        self.withdraw_document(doc_id, version)
    }
    fn sink_revision(&self) -> Rev {
        self.revision()
    }
}

/// The two revisions of a world: after its base sources (phase "before") and after its edit stream (phase "after").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorldRevs {
    pub base: Rev,
    pub final_: Rev,
}

/// xr_codecs.world_store, op for op (so revisions, and exact fingerprints, equal the reference's).
pub fn ingest_world<S: WriteSink + ?Sized>(
    sink: &mut S,
    w: &World<'_>,
) -> Result<WorldRevs, ContractError> {
    for d in &w.base_documents {
        sink.sink_add_document(&d.doc_id, &d.version, &d.text)?;
    }
    for f in &w.base_facts {
        let fo = f.to_fact_in();
        sink.sink_add_fact(&fo.get(), &[])?;
    }
    let base = sink.sink_revision();
    for e in &w.edits {
        match &*e.op {
            "add_document" | "correct" | "add_conflict" => {
                let d = e
                    .doc
                    .as_ref()
                    .ok_or(ContractError::new("edit without doc"))?;
                sink.sink_add_document(&d.doc_id, &d.version, &d.text)?;
                match &*e.op {
                    "add_document" => {
                        for f in e.facts.iter().flatten() {
                            let fo = f.to_fact_in();
                            sink.sink_add_fact(&fo.get(), &[])?;
                        }
                    }
                    "correct" => {
                        let f = e
                            .fact
                            .as_ref()
                            .ok_or(ContractError::new("correct without fact"))?;
                        let sup = e
                            .supersedes
                            .as_deref()
                            .ok_or(ContractError::new("correct without supersedes"))?;
                        let fo = f.to_fact_in();
                        sink.sink_add_fact(&fo.get(), &[sup])?;
                    }
                    _ => {
                        let f = e
                            .fact
                            .as_ref()
                            .ok_or(ContractError::new("add_conflict without fact"))?;
                        let fo = f.to_fact_in();
                        sink.sink_add_fact(&fo.get(), &[])?;
                    }
                }
            }
            "withdraw_document" => {
                let (d, v) = (e.doc_id.as_deref(), e.version.as_deref());
                sink.sink_withdraw_document(d.unwrap_or(""), v.unwrap_or(""))?;
            }
            other => return Err(ContractError::owned(format!("unknown edit {other}"))),
        }
    }
    Ok(WorldRevs {
        base,
        final_: sink.sink_revision(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{Status, Store};

    const LINE: &str = r#"{"schema": "memory-world-v1", "group_id": "g", "base_documents": [{"doc_id": "g/doc1", "version": "v1", "text": "Ana Bel was born on 3 February 1984. This person is 170 cm tall."}], "base_facts": [{"fact_id": "g/doc1@v1/birth_date", "subject": "Ana Bel", "relation": "birth_date", "kind": "date", "value": "1984-02-03", "unit": "", "doc_id": "g/doc1", "version": "v1", "start": 20, "end": 35, "quote": "3 February 1984", "scope": "default"}, {"fact_id": "g/doc1@v1/height", "subject": "Ana Bel", "relation": "height", "kind": "number", "value": "170", "unit": "cm", "doc_id": "g/doc1", "version": "v1", "start": 52, "end": 55, "quote": "170", "scope": "default"}], "edits": [{"op": "correct", "doc": {"doc_id": "g/doc1", "version": "v2", "text": "Ana Bel is 171 cm tall."}, "fact": {"fact_id": "g/doc1@v2/height", "subject": "Ana Bel", "relation": "height", "kind": "number", "value": "171", "unit": "cm", "doc_id": "g/doc1", "version": "v2", "start": 11, "end": 14, "quote": "171", "scope": "default"}, "supersedes": "g/doc1@v1/height"}], "items": [{"item_id": "g/x", "question": "What is the height of Ana Bel?", "phase": "after"}]}"#;

    #[test]
    fn ingest_world_order() {
        let w = World::parse(LINE).unwrap();
        let mut s = Store::default();
        let revs = ingest_world(&mut s, &w).unwrap();
        assert_eq!(revs, WorldRevs { base: 3, final_: 5 });
        let va = crate::store::ValidAt::parse(crate::store::VALID_AT).unwrap();
        let before = s.at(revs.base, va).lookup("Ana Bel", "height", "default");
        let after = s.at(revs.final_, va).lookup("Ana Bel", "height", "default");
        assert_eq!(before.value.unwrap().value, "17E1");
        assert_eq!(after.value.unwrap().value, "171E0");
        assert_eq!(after.status, Status::Value);
        assert!(matches!(w.base_documents[0].text, Cow::Borrowed(_)));
    }
}
