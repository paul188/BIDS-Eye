-- The schema as it stood on 2026-11-25, before Alembic: the models of BIDS-SQL
-- and backend/models.py, plus the fourteen crawlers/migrate_add_*.sql files.
-- Written by pg_dump --schema-only from a database built that way.
-- Run by 0001_baseline.py, one statement at a time.

CREATE EXTENSION IF NOT EXISTS pg_trgm WITH SCHEMA public;

CREATE TABLE public.bids_dataset_facets (
    dataset_id uuid NOT NULL,
    datatype text,
    suffix character varying,
    task character varying
);

CREATE TABLE public.bids_dataset_facts (
    id bigint NOT NULL,
    dataset_id uuid NOT NULL,
    kind text NOT NULL,
    term text NOT NULL,
    role text,
    code text,
    score real,
    evidence text,
    verified boolean NOT NULL,
    CONSTRAINT bids_dataset_facts_kind_check CHECK ((kind = ANY (ARRAY['condition'::text, 'paradigm'::text])))
);

CREATE SEQUENCE public.bids_dataset_facts_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

ALTER SEQUENCE public.bids_dataset_facts_id_seq OWNED BY public.bids_dataset_facts.id;

CREATE TABLE public.bids_dataset_papers (
    dataset_id uuid NOT NULL,
    paper_id bigint NOT NULL,
    role text NOT NULL,
    role_reason text,
    sources text[] NOT NULL,
    raw_ref text,
    "position" integer DEFAULT 0 NOT NULL,
    CONSTRAINT bids_dataset_papers_role_check CHECK ((role = ANY (ARRAY['primary'::text, 'related'::text, 'tool'::text, 'reuse'::text])))
);

CREATE TABLE public.bids_dataset_sites (
    id bigint NOT NULL,
    dataset_id uuid NOT NULL,
    name_raw text NOT NULL,
    source text NOT NULL,
    ror_id text,
    evidence text
);

CREATE SEQUENCE public.bids_dataset_sites_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

ALTER SEQUENCE public.bids_dataset_sites_id_seq OWNED BY public.bids_dataset_sites.id;

CREATE TABLE public.bids_datasets (
    id uuid NOT NULL,
    root_path character varying NOT NULL,
    description json NOT NULL,
    name character varying(100) NOT NULL,
    bids_version character varying(20) NOT NULL,
    dataset_type character varying(20) NOT NULL,
    parent_id uuid,
    source_type character varying(20) NOT NULL,
    accession_id character varying(50),
    remote_url character varying,
    authors character varying[],
    license text,
    doi character varying,
    paper_references character varying[],
    funding character varying[],
    description_text text,
    validation_status character varying(20),
    validation_errors json,
    validation_warnings json,
    validator_version character varying(50),
    validated_at timestamp with time zone,
    readme_text text,
    dx_status text,
    study_domain text,
    study_design text,
    species text,
    study_longitudinal text,
    senior_author text,
    publish_date timestamp with time zone,
    on_brainlife boolean,
    downloads integer,
    views integer,
    age_min double precision,
    age_max double precision,
    size_bytes bigint,
    deprecated_reason text,
    institutions text[],
    llm_profile text,
    llm_conditions text,
    llm_profile_meta jsonb,
    search_tsv tsvector GENERATED ALWAYS AS ((((setweight(to_tsvector('english'::regconfig, translate((COALESCE(name, ''::character varying))::text, '/'::text, ' '::text)), 'A'::"char") || setweight(to_tsvector('english'::regconfig, translate(((((((((COALESCE(dx_status, ''::text) || ' '::text) || COALESCE(study_domain, ''::text)) || ' '::text) || COALESCE(study_design, ''::text)) || ' '::text) || COALESCE(description_text, ''::text)) || ' '::text) || COALESCE(llm_conditions, ''::text)), '/'::text, ' '::text)), 'B'::"char")) || setweight(to_tsvector('english'::regconfig, translate(COALESCE(llm_profile, ''::text), '/'::text, ' '::text)), 'C'::"char")) || setweight(to_tsvector('english'::regconfig, translate(COALESCE(readme_text, ''::text), '/'::text, ' '::text)), 'D'::"char"))) STORED,
    paper_text text,
    papers_checked_at timestamp with time zone,
    search_tsv_papers tsvector GENERATED ALWAYS AS (((((setweight(to_tsvector('english'::regconfig, translate((COALESCE(name, ''::character varying))::text, '/'::text, ' '::text)), 'A'::"char") || setweight(to_tsvector('english'::regconfig, translate(((((((((COALESCE(dx_status, ''::text) || ' '::text) || COALESCE(study_domain, ''::text)) || ' '::text) || COALESCE(study_design, ''::text)) || ' '::text) || COALESCE(description_text, ''::text)) || ' '::text) || COALESCE(llm_conditions, ''::text)), '/'::text, ' '::text)), 'B'::"char")) || setweight(to_tsvector('english'::regconfig, translate(COALESCE(llm_profile, ''::text), '/'::text, ' '::text)), 'C'::"char")) || setweight(to_tsvector('english'::regconfig, translate(COALESCE(readme_text, ''::text), '/'::text, ' '::text)), 'D'::"char")) || setweight(to_tsvector('english'::regconfig, translate(COALESCE(paper_text, ''::text), '/'::text, ' '::text)), 'D'::"char"))) STORED,
    search_tsv_exact tsvector GENERATED ALWAYS AS ((((setweight(to_tsvector('simple'::regconfig, translate((COALESCE(name, ''::character varying))::text, '/'::text, ' '::text)), 'A'::"char") || setweight(to_tsvector('simple'::regconfig, translate(((((((((COALESCE(dx_status, ''::text) || ' '::text) || COALESCE(study_domain, ''::text)) || ' '::text) || COALESCE(study_design, ''::text)) || ' '::text) || COALESCE(description_text, ''::text)) || ' '::text) || COALESCE(llm_conditions, ''::text)), '/'::text, ' '::text)), 'B'::"char")) || setweight(to_tsvector('simple'::regconfig, translate(COALESCE(llm_profile, ''::text), '/'::text, ' '::text)), 'C'::"char")) || setweight(to_tsvector('simple'::regconfig, translate(COALESCE(readme_text, ''::text), '/'::text, ' '::text)), 'D'::"char"))) STORED,
    search_tsv_exact_papers tsvector GENERATED ALWAYS AS (((((setweight(to_tsvector('simple'::regconfig, translate((COALESCE(name, ''::character varying))::text, '/'::text, ' '::text)), 'A'::"char") || setweight(to_tsvector('simple'::regconfig, translate(((((((((COALESCE(dx_status, ''::text) || ' '::text) || COALESCE(study_domain, ''::text)) || ' '::text) || COALESCE(study_design, ''::text)) || ' '::text) || COALESCE(description_text, ''::text)) || ' '::text) || COALESCE(llm_conditions, ''::text)), '/'::text, ' '::text)), 'B'::"char")) || setweight(to_tsvector('simple'::regconfig, translate(COALESCE(llm_profile, ''::text), '/'::text, ' '::text)), 'C'::"char")) || setweight(to_tsvector('simple'::regconfig, translate(COALESCE(readme_text, ''::text), '/'::text, ' '::text)), 'D'::"char")) || setweight(to_tsvector('simple'::regconfig, translate(COALESCE(paper_text, ''::text), '/'::text, ' '::text)), 'D'::"char"))) STORED,
    n_subjects integer,
    paper_relation jsonb,
    dataset_summary text,
    dataset_summary_meta jsonb
);

CREATE TABLE public.bids_derivatives (
    id bigint NOT NULL,
    dataset_id uuid NOT NULL,
    pipeline text NOT NULL,
    pipeline_name text,
    pipeline_version text,
    source text NOT NULL,
    location text NOT NULL,
    n_subjects integer,
    fetched_at timestamp with time zone DEFAULT now() NOT NULL
);

CREATE SEQUENCE public.bids_derivatives_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

ALTER SEQUENCE public.bids_derivatives_id_seq OWNED BY public.bids_derivatives.id;

CREATE TABLE public.bids_objects (
    id uuid NOT NULL,
    path character varying NOT NULL,
    subject character varying,
    subject_index integer,
    session character varying,
    session_index integer,
    task character varying,
    run integer,
    suffix character varying NOT NULL,
    datatype character varying(64),
    extension character varying(32),
    other_entities json NOT NULL,
    is_remote boolean NOT NULL,
    dataset_id uuid NOT NULL,
    class_ character varying(20) NOT NULL,
    filename character varying NOT NULL
);

CREATE TABLE public.bids_papers (
    id bigint NOT NULL,
    paper_key text NOT NULL,
    doi text,
    preprint_doi text,
    pmid text,
    pmcid text,
    arxiv_id text,
    title text NOT NULL,
    abstract text,
    year integer,
    journal text,
    is_preprint boolean DEFAULT false NOT NULL,
    authors text[],
    mesh_terms text[],
    mesh_major text[],
    keywords text[],
    is_open_access boolean,
    license text,
    source text NOT NULL,
    fetched_at timestamp with time zone DEFAULT now() NOT NULL
);

CREATE SEQUENCE public.bids_papers_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

ALTER SEQUENCE public.bids_papers_id_seq OWNED BY public.bids_papers.id;

CREATE TABLE public.bids_participants (
    id uuid NOT NULL,
    dataset_id uuid NOT NULL,
    participant_id character varying(50) NOT NULL,
    age double precision,
    sex character varying(10),
    handedness character varying(10),
    diagnosis character varying(200),
    extra json
);

CREATE TABLE public.bids_site_places (
    name_raw text NOT NULL,
    ror_id text DEFAULT ''::text NOT NULL,
    institution text,
    city text,
    region text,
    country text,
    country_code text,
    lat double precision,
    lng double precision,
    matched_at timestamp with time zone DEFAULT now() NOT NULL
);

CREATE TABLE public.bids_word_forms (
    form text NOT NULL,
    stem text NOT NULL,
    ndoc integer NOT NULL
);

CREATE TABLE public.object_associations (
    src_id uuid NOT NULL,
    dst_id uuid NOT NULL,
    kind character varying(50) NOT NULL
);

CREATE TABLE public.query_interpretations (
    key character varying(64) NOT NULL,
    pipeline_version character varying(16) NOT NULL,
    question text NOT NULL,
    question_normalized text NOT NULL,
    payload text NOT NULL,
    total_at_creation integer NOT NULL,
    hits integer DEFAULT 0 NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    last_used_at timestamp with time zone,
    review_verdict character varying(16),
    review_notes text,
    reviewed_at timestamp with time zone
);

CREATE TABLE public.query_log (
    id uuid NOT NULL,
    asked_at timestamp with time zone DEFAULT now() NOT NULL,
    question text NOT NULL,
    pipeline_version character varying(16) NOT NULL,
    payload text NOT NULL,
    total integer NOT NULL,
    from_cache boolean NOT NULL,
    relaxed boolean NOT NULL,
    self_corrected boolean NOT NULL,
    fallback boolean NOT NULL
);

CREATE TABLE public.search_history (
    id uuid NOT NULL,
    github_id bigint NOT NULL,
    asked_at timestamp with time zone DEFAULT now() NOT NULL,
    question text NOT NULL,
    total integer NOT NULL,
    answer text NOT NULL
);

CREATE TABLE public.search_history_consents (
    github_id bigint NOT NULL,
    consented_at timestamp with time zone DEFAULT now() NOT NULL,
    last_used_at timestamp with time zone DEFAULT now() NOT NULL
);

CREATE SEQUENCE public.search_history_consents_github_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

ALTER SEQUENCE public.search_history_consents_github_id_seq OWNED BY public.search_history_consents.github_id;

ALTER TABLE ONLY public.bids_dataset_facts ALTER COLUMN id SET DEFAULT nextval('public.bids_dataset_facts_id_seq'::regclass);

ALTER TABLE ONLY public.bids_dataset_sites ALTER COLUMN id SET DEFAULT nextval('public.bids_dataset_sites_id_seq'::regclass);

ALTER TABLE ONLY public.bids_derivatives ALTER COLUMN id SET DEFAULT nextval('public.bids_derivatives_id_seq'::regclass);

ALTER TABLE ONLY public.bids_papers ALTER COLUMN id SET DEFAULT nextval('public.bids_papers_id_seq'::regclass);

ALTER TABLE ONLY public.search_history_consents ALTER COLUMN github_id SET DEFAULT nextval('public.search_history_consents_github_id_seq'::regclass);

ALTER TABLE ONLY public.bids_dataset_facts
    ADD CONSTRAINT bids_dataset_facts_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.bids_dataset_papers
    ADD CONSTRAINT bids_dataset_papers_pkey PRIMARY KEY (dataset_id, paper_id);

ALTER TABLE ONLY public.bids_dataset_sites
    ADD CONSTRAINT bids_dataset_sites_dataset_id_source_name_raw_key UNIQUE (dataset_id, source, name_raw);

ALTER TABLE ONLY public.bids_dataset_sites
    ADD CONSTRAINT bids_dataset_sites_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.bids_datasets
    ADD CONSTRAINT bids_datasets_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.bids_derivatives
    ADD CONSTRAINT bids_derivatives_dataset_id_source_location_key UNIQUE (dataset_id, source, location);

ALTER TABLE ONLY public.bids_derivatives
    ADD CONSTRAINT bids_derivatives_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.bids_objects
    ADD CONSTRAINT bids_objects_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.bids_papers
    ADD CONSTRAINT bids_papers_paper_key_key UNIQUE (paper_key);

ALTER TABLE ONLY public.bids_papers
    ADD CONSTRAINT bids_papers_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.bids_participants
    ADD CONSTRAINT bids_participants_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.bids_site_places
    ADD CONSTRAINT bids_site_places_pkey PRIMARY KEY (name_raw, ror_id);

ALTER TABLE ONLY public.bids_word_forms
    ADD CONSTRAINT bids_word_forms_pkey PRIMARY KEY (form);

ALTER TABLE ONLY public.object_associations
    ADD CONSTRAINT object_associations_pkey PRIMARY KEY (src_id, dst_id, kind);

ALTER TABLE ONLY public.query_interpretations
    ADD CONSTRAINT query_interpretations_pkey PRIMARY KEY (key);

ALTER TABLE ONLY public.query_log
    ADD CONSTRAINT query_log_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.search_history_consents
    ADD CONSTRAINT search_history_consents_pkey PRIMARY KEY (github_id);

ALTER TABLE ONLY public.search_history
    ADD CONSTRAINT search_history_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.bids_participants
    ADD CONSTRAINT uq_participant_per_dataset UNIQUE (dataset_id, participant_id);

CREATE INDEX idx_bids_dataset_facts_code ON public.bids_dataset_facts USING btree (kind, code) WHERE verified;

CREATE INDEX idx_bids_dataset_facts_dataset ON public.bids_dataset_facts USING btree (dataset_id);

CREATE INDEX idx_bids_dataset_papers_paper ON public.bids_dataset_papers USING btree (paper_id);

CREATE INDEX idx_bids_dataset_sites_dataset ON public.bids_dataset_sites USING btree (dataset_id);

CREATE INDEX idx_bids_dataset_sites_name_raw ON public.bids_dataset_sites USING btree (name_raw);

CREATE INDEX idx_bids_dataset_sites_source ON public.bids_dataset_sites USING btree (source);

CREATE INDEX idx_bids_datasets_description_text_fts ON public.bids_datasets USING gin (to_tsvector('english'::regconfig, COALESCE(description_text, ''::text)));

CREATE INDEX idx_bids_datasets_name_fts ON public.bids_datasets USING gin (to_tsvector('english'::regconfig, (COALESCE(name, ''::character varying))::text));

CREATE INDEX idx_bids_datasets_search_tsv ON public.bids_datasets USING gin (search_tsv);

CREATE INDEX idx_bids_datasets_search_tsv_exact ON public.bids_datasets USING gin (search_tsv_exact);

CREATE INDEX idx_bids_datasets_search_tsv_exact_papers ON public.bids_datasets USING gin (search_tsv_exact_papers);

CREATE INDEX idx_bids_datasets_search_tsv_papers ON public.bids_datasets USING gin (search_tsv_papers);

CREATE INDEX idx_bids_derivatives_dataset ON public.bids_derivatives USING btree (dataset_id);

CREATE INDEX idx_bids_derivatives_pipeline ON public.bids_derivatives USING btree (pipeline);

CREATE INDEX idx_bids_site_places_city ON public.bids_site_places USING btree (city);

CREATE INDEX idx_bids_site_places_country ON public.bids_site_places USING btree (country);

CREATE INDEX idx_bids_site_places_region ON public.bids_site_places USING btree (region);

CREATE INDEX idx_bids_word_forms_stem ON public.bids_word_forms USING btree (stem);

CREATE INDEX ix_bids_dataset_facets_dataset ON public.bids_dataset_facets USING btree (dataset_id);

CREATE INDEX ix_bids_dataset_facets_datatype ON public.bids_dataset_facets USING btree (datatype);

CREATE INDEX ix_bids_dataset_facets_suffix ON public.bids_dataset_facets USING btree (suffix);

CREATE INDEX ix_bids_dataset_facets_task ON public.bids_dataset_facets USING btree (task);

CREATE INDEX ix_bids_datasets_accession_id ON public.bids_datasets USING btree (accession_id);

CREATE INDEX ix_bids_datasets_authors ON public.bids_datasets USING gin (authors);

CREATE INDEX ix_bids_datasets_dataset_type ON public.bids_datasets USING btree (dataset_type);

CREATE INDEX ix_bids_datasets_doi ON public.bids_datasets USING btree (doi);

CREATE INDEX ix_bids_datasets_funding ON public.bids_datasets USING gin (funding);

CREATE INDEX ix_bids_datasets_institutions ON public.bids_datasets USING gin (institutions);

CREATE INDEX ix_bids_datasets_license ON public.bids_datasets USING btree (license);

CREATE INDEX ix_bids_datasets_name ON public.bids_datasets USING btree (name);

CREATE INDEX ix_bids_datasets_parent_id ON public.bids_datasets USING btree (parent_id);

CREATE INDEX ix_bids_datasets_references ON public.bids_datasets USING gin (paper_references);

CREATE UNIQUE INDEX ix_bids_datasets_root_path ON public.bids_datasets USING btree (root_path);

CREATE INDEX ix_bids_datasets_source_type ON public.bids_datasets USING btree (source_type);

CREATE INDEX ix_bids_objects_dataset_filetype ON public.bids_objects USING btree (dataset_id, datatype, suffix, task);

CREATE INDEX ix_bids_objects_dataset_id ON public.bids_objects USING btree (dataset_id);

CREATE INDEX ix_bids_objects_dataset_subject ON public.bids_objects USING btree (dataset_id, subject);

CREATE INDEX ix_bids_objects_datatype ON public.bids_objects USING btree (datatype);

CREATE INDEX ix_bids_objects_extension ON public.bids_objects USING btree (extension);

CREATE INDEX ix_bids_objects_field_strength ON public.bids_objects USING btree (dataset_id, (
CASE
    WHEN ((other_entities ->> 'MagneticFieldStrength'::text) ~ '^[[:space:]]*[-+]?([0-9]+[.]?[0-9]*|[.][0-9]+)([eE][-+]?[0-9]+)?[[:space:]]*$'::text) THEN ((other_entities ->> 'MagneticFieldStrength'::text))::double precision
    ELSE NULL::double precision
END)) WHERE (
CASE
    WHEN ((other_entities ->> 'MagneticFieldStrength'::text) ~ '^[[:space:]]*[-+]?([0-9]+[.]?[0-9]*|[.][0-9]+)([eE][-+]?[0-9]+)?[[:space:]]*$'::text) THEN ((other_entities ->> 'MagneticFieldStrength'::text))::double precision
    ELSE NULL::double precision
END IS NOT NULL);

CREATE INDEX ix_bids_objects_is_remote ON public.bids_objects USING btree (is_remote);

CREATE INDEX ix_bids_objects_manufacturer ON public.bids_objects USING btree (dataset_id, ((other_entities ->> 'Manufacturer'::text))) WHERE ((other_entities ->> 'Manufacturer'::text) IS NOT NULL);

CREATE INDEX ix_bids_objects_manufacturer_trgm ON public.bids_objects USING gin (((other_entities ->> 'Manufacturer'::text)) public.gin_trgm_ops) WHERE ((other_entities ->> 'Manufacturer'::text) IS NOT NULL);

CREATE UNIQUE INDEX ix_bids_objects_path ON public.bids_objects USING btree (path);

CREATE INDEX ix_bids_objects_repetition_time ON public.bids_objects USING btree (dataset_id, (
CASE
    WHEN ((other_entities ->> 'RepetitionTime'::text) ~ '^[[:space:]]*[-+]?([0-9]+[.]?[0-9]*|[.][0-9]+)([eE][-+]?[0-9]+)?[[:space:]]*$'::text) THEN ((other_entities ->> 'RepetitionTime'::text))::double precision
    ELSE NULL::double precision
END)) WHERE (
CASE
    WHEN ((other_entities ->> 'RepetitionTime'::text) ~ '^[[:space:]]*[-+]?([0-9]+[.]?[0-9]*|[.][0-9]+)([eE][-+]?[0-9]+)?[[:space:]]*$'::text) THEN ((other_entities ->> 'RepetitionTime'::text))::double precision
    ELSE NULL::double precision
END IS NOT NULL);

CREATE INDEX ix_bids_objects_run ON public.bids_objects USING btree (run);

CREATE INDEX ix_bids_objects_sampling_frequency ON public.bids_objects USING btree (dataset_id, (
CASE
    WHEN ((other_entities ->> 'SamplingFrequency'::text) ~ '^[[:space:]]*[-+]?([0-9]+[.]?[0-9]*|[.][0-9]+)([eE][-+]?[0-9]+)?[[:space:]]*$'::text) THEN ((other_entities ->> 'SamplingFrequency'::text))::double precision
    ELSE NULL::double precision
END)) WHERE (
CASE
    WHEN ((other_entities ->> 'SamplingFrequency'::text) ~ '^[[:space:]]*[-+]?([0-9]+[.]?[0-9]*|[.][0-9]+)([eE][-+]?[0-9]+)?[[:space:]]*$'::text) THEN ((other_entities ->> 'SamplingFrequency'::text))::double precision
    ELSE NULL::double precision
END IS NOT NULL);

CREATE INDEX ix_bids_objects_scanner_model_any_trgm ON public.bids_objects USING gin (COALESCE((other_entities ->> 'ManufacturersModelName'::text), (other_entities ->> 'ManufacturerModelName'::text)) public.gin_trgm_ops) WHERE (COALESCE((other_entities ->> 'ManufacturersModelName'::text), (other_entities ->> 'ManufacturerModelName'::text)) IS NOT NULL);

CREATE INDEX ix_bids_objects_session ON public.bids_objects USING btree (session);

CREATE INDEX ix_bids_objects_session_index ON public.bids_objects USING btree (session_index);

CREATE INDEX ix_bids_objects_subject ON public.bids_objects USING btree (subject);

CREATE INDEX ix_bids_objects_subject_index ON public.bids_objects USING btree (subject_index);

CREATE INDEX ix_bids_objects_suffix ON public.bids_objects USING btree (suffix);

CREATE INDEX ix_bids_objects_task ON public.bids_objects USING btree (task);

CREATE INDEX ix_bids_participants_dataset_id ON public.bids_participants USING btree (dataset_id);

CREATE INDEX ix_bids_participants_diagnosis ON public.bids_participants USING btree (diagnosis);

CREATE INDEX ix_bids_participants_participant_id ON public.bids_participants USING btree (participant_id);

CREATE INDEX ix_bids_participants_sex ON public.bids_participants USING btree (sex);

CREATE INDEX ix_query_interpretations_created_at ON public.query_interpretations USING btree (created_at);

CREATE INDEX ix_query_interpretations_pipeline_version ON public.query_interpretations USING btree (pipeline_version);

CREATE INDEX ix_query_interpretations_question_normalized ON public.query_interpretations USING btree (question_normalized);

CREATE INDEX ix_query_log_asked_at ON public.query_log USING btree (asked_at);

CREATE INDEX ix_query_log_pipeline_version ON public.query_log USING btree (pipeline_version);

CREATE INDEX ix_search_history_asked_at ON public.search_history USING btree (asked_at);

CREATE INDEX ix_search_history_consents_last_used_at ON public.search_history_consents USING btree (last_used_at);

CREATE INDEX ix_search_history_github_id ON public.search_history USING btree (github_id);

ALTER TABLE ONLY public.bids_dataset_facets
    ADD CONSTRAINT bids_dataset_facets_dataset_id_fkey FOREIGN KEY (dataset_id) REFERENCES public.bids_datasets(id) ON DELETE CASCADE;

ALTER TABLE ONLY public.bids_dataset_facts
    ADD CONSTRAINT bids_dataset_facts_dataset_id_fkey FOREIGN KEY (dataset_id) REFERENCES public.bids_datasets(id) ON DELETE CASCADE;

ALTER TABLE ONLY public.bids_dataset_papers
    ADD CONSTRAINT bids_dataset_papers_dataset_id_fkey FOREIGN KEY (dataset_id) REFERENCES public.bids_datasets(id) ON DELETE CASCADE;

ALTER TABLE ONLY public.bids_dataset_papers
    ADD CONSTRAINT bids_dataset_papers_paper_id_fkey FOREIGN KEY (paper_id) REFERENCES public.bids_papers(id) ON DELETE CASCADE;

ALTER TABLE ONLY public.bids_dataset_sites
    ADD CONSTRAINT bids_dataset_sites_dataset_id_fkey FOREIGN KEY (dataset_id) REFERENCES public.bids_datasets(id) ON DELETE CASCADE;

ALTER TABLE ONLY public.bids_datasets
    ADD CONSTRAINT bids_datasets_parent_id_fkey FOREIGN KEY (parent_id) REFERENCES public.bids_datasets(id) ON DELETE SET NULL;

ALTER TABLE ONLY public.bids_derivatives
    ADD CONSTRAINT bids_derivatives_dataset_id_fkey FOREIGN KEY (dataset_id) REFERENCES public.bids_datasets(id) ON DELETE CASCADE;

ALTER TABLE ONLY public.bids_objects
    ADD CONSTRAINT bids_objects_dataset_id_fkey FOREIGN KEY (dataset_id) REFERENCES public.bids_datasets(id);

ALTER TABLE ONLY public.bids_participants
    ADD CONSTRAINT bids_participants_dataset_id_fkey FOREIGN KEY (dataset_id) REFERENCES public.bids_datasets(id) ON DELETE CASCADE;

ALTER TABLE ONLY public.object_associations
    ADD CONSTRAINT object_associations_dst_id_fkey FOREIGN KEY (dst_id) REFERENCES public.bids_objects(id) ON DELETE CASCADE;

ALTER TABLE ONLY public.object_associations
    ADD CONSTRAINT object_associations_src_id_fkey FOREIGN KEY (src_id) REFERENCES public.bids_objects(id) ON DELETE CASCADE;

ALTER TABLE ONLY public.search_history
    ADD CONSTRAINT search_history_github_id_fkey FOREIGN KEY (github_id) REFERENCES public.search_history_consents(github_id) ON DELETE CASCADE;
