//! Lecture du corpus : les documents d'un fichier Parquet de FineWeb-Edu,
//! et la règle qui les répartit entre entraînement et validation.

use std::error::Error;
use std::fs::File;
use std::path::Path;

use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use parquet::schema::types::Type;

/// Un document sur `VAL_EVERY` part en validation, soit 5 %.
const VAL_EVERY: u64 = 20;

/// Le découpage se fait par document entier, jamais au milieu d'un texte,
/// et ne dépend que du rang du document dans son fichier : il est le même
/// pour tous les programmes et tous les tokeniseurs.
pub fn is_validation(document_index: u64) -> bool {
    document_index.is_multiple_of(VAL_EVERY)
}

/// La colonne `text` d'un fichier Parquet, lue un groupe de lignes à la
/// fois : seul le groupe courant est en mémoire.
pub struct ParquetTexts {
    reader: SerializedFileReader<File>,
    /// Schéma réduit à la seule colonne `text`. Le lecteur ne décompresse
    /// alors pas les autres colonnes (url, score, etc.).
    projection: Type,
}

impl ParquetTexts {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Box<dyn Error>> {
        let reader = SerializedFileReader::new(File::open(path)?)?;
        let schema = reader.metadata().file_metadata().schema();
        let text_column = schema
            .get_fields()
            .iter()
            .find(|field| field.name() == "text")
            .ok_or("colonne `text` absente du fichier")?
            .clone();
        let projection = Type::group_type_builder(schema.name())
            .with_fields(vec![text_column])
            .build()?;
        Ok(Self { reader, projection })
    }

    pub fn documents(&self) -> u64 {
        self.reader.metadata().file_metadata().num_rows() as u64
    }

    pub fn row_groups(&self) -> usize {
        self.reader.num_row_groups()
    }

    /// Les textes d'un groupe de lignes, dans l'ordre du fichier.
    pub fn read_group(&self, index: usize) -> Result<Vec<String>, Box<dyn Error>> {
        let group = self.reader.get_row_group(index)?;
        let mut texts = Vec::with_capacity(group.metadata().num_rows() as usize);
        for row in group.get_row_iter(Some(self.projection.clone()))? {
            texts.push(row?.get_string(0)?.clone());
        }
        Ok(texts)
    }
}
