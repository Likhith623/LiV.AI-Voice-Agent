

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;


CREATE SCHEMA IF NOT EXISTS "public";


ALTER SCHEMA "public" OWNER TO "pg_database_owner";


COMMENT ON SCHEMA "public" IS 'standard public schema';



CREATE OR REPLACE FUNCTION "public"."auto_generate_slug"() RETURNS "trigger"
    LANGUAGE "plpgsql"
    AS $$
BEGIN
    IF NEW.slug IS NULL OR NEW.slug = '' THEN
        NEW.slug := generate_slug(NEW.title);
    END IF;
    RETURN NEW;
END;
$$;


ALTER FUNCTION "public"."auto_generate_slug"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."cleanup_old_drafts"("days_old" integer DEFAULT 30) RETURNS integer
    LANGUAGE "plpgsql"
    AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM blog_posts 
    WHERE status = 'draft' 
    AND created_at < NOW() - INTERVAL '1 day' * days_old;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$;


ALTER FUNCTION "public"."cleanup_old_drafts"("days_old" integer) OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."generate_slug"("title" "text") RETURNS "text"
    LANGUAGE "plpgsql"
    AS $$
BEGIN
    RETURN lower(regexp_replace(regexp_replace(title, '[^a-zA-Z0-9\s-]', '', 'g'), '\s+', '-', 'g'));
END;
$$;


ALTER FUNCTION "public"."generate_slug"("title" "text") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."get_blog_stats"() RETURNS TABLE("total_posts" bigint, "published_posts" bigint, "draft_posts" bigint, "archived_posts" bigint, "total_views" bigint, "unique_authors" bigint)
    LANGUAGE "plpgsql"
    AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*) as total_posts,
        COUNT(*) FILTER (WHERE status = 'published') as published_posts,
        COUNT(*) FILTER (WHERE status = 'draft') as draft_posts,
        COUNT(*) FILTER (WHERE status = 'archived') as archived_posts,
        COALESCE(SUM(view_count), 0) as total_views,
        COUNT(DISTINCT author_email) as unique_authors
    FROM blog_posts;
END;
$$;


ALTER FUNCTION "public"."get_blog_stats"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."increment_view_count"("post_id" "uuid") RETURNS boolean
    LANGUAGE "plpgsql"
    AS $$
BEGIN
    UPDATE blog_posts 
    SET view_count = view_count + 1 
    WHERE id = post_id;
    
    RETURN FOUND;
END;
$$;


ALTER FUNCTION "public"."increment_view_count"("post_id" "uuid") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."search_blog_posts"("search_term" "text" DEFAULT NULL::"text", "tag_filter" "text" DEFAULT NULL::"text", "author_filter" "text" DEFAULT NULL::"text", "status_filter" "text" DEFAULT 'published'::"text", "page_num" integer DEFAULT 1, "page_size" integer DEFAULT 10) RETURNS TABLE("id" "uuid", "title" "text", "slug" "text", "content" "text", "excerpt" "text", "tags" "text"[], "publish_date" "date", "status" "text", "author_email" "text", "featured_image_url" "text", "additional_images" "text"[], "view_count" integer, "created_at" timestamp with time zone, "updated_at" timestamp with time zone, "total_count" bigint)
    LANGUAGE "plpgsql"
    AS $$
DECLARE
    offset_val INTEGER;
BEGIN
    offset_val := (page_num - 1) * page_size;
    
    RETURN QUERY
    WITH filtered_posts AS (
        SELECT 
            bp.*,
            COUNT(*) OVER() as total_count
        FROM blog_posts bp
        WHERE 
            (search_term IS NULL OR 
             bp.title ILIKE '%' || search_term || '%' OR
             bp.content ILIKE '%' || search_term || '%' OR
             bp.excerpt ILIKE '%' || search_term || '%' OR
             search_term = ANY(bp.tags))
        AND (tag_filter IS NULL OR tag_filter = ANY(bp.tags))
        AND (author_filter IS NULL OR bp.author_email = author_filter)
        AND (status_filter IS NULL OR bp.status = status_filter)
        ORDER BY bp.created_at DESC
        LIMIT page_size OFFSET offset_val
    )
    SELECT 
        fp.id,
        fp.title,
        fp.slug,
        fp.content,
        fp.excerpt,
        fp.tags,
        fp.publish_date,
        fp.status,
        fp.author_email,
        fp.featured_image_url,
        fp.additional_images,
        fp.view_count,
        fp.created_at,
        fp.updated_at,
        fp.total_count
    FROM filtered_posts fp;
END;
$$;


ALTER FUNCTION "public"."search_blog_posts"("search_term" "text", "tag_filter" "text", "author_filter" "text", "status_filter" "text", "page_num" integer, "page_size" integer) OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."set_blog_post_slug"() RETURNS "trigger"
    LANGUAGE "plpgsql"
    AS $$
BEGIN
    IF NEW.slug IS NULL OR NEW.slug = '' THEN
        NEW.slug := generate_slug(NEW.title);
    END IF;
    RETURN NEW;
END;
$$;


ALTER FUNCTION "public"."set_blog_post_slug"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."update_updated_at_column"() RETURNS "trigger"
    LANGUAGE "plpgsql"
    AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;


ALTER FUNCTION "public"."update_updated_at_column"() OWNER TO "postgres";

SET default_tablespace = '';

SET default_table_access_method = "heap";


CREATE TABLE IF NOT EXISTS "public"."blog_posts" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "title" "text" NOT NULL,
    "slug" "text" NOT NULL,
    "content" "text" NOT NULL,
    "excerpt" "text",
    "tags" "text"[] DEFAULT '{}'::"text"[],
    "publish_date" "date" DEFAULT CURRENT_DATE NOT NULL,
    "status" "text" DEFAULT 'draft'::"text" NOT NULL,
    "author_email" "text" NOT NULL,
    "featured_image_url" "text",
    "additional_images" "text"[] DEFAULT '{}'::"text"[],
    "view_count" integer DEFAULT 0,
    "created_at" timestamp with time zone DEFAULT "now"(),
    "updated_at" timestamp with time zone DEFAULT "now"(),
    CONSTRAINT "blog_posts_status_check" CHECK (("status" = ANY (ARRAY['draft'::"text", 'published'::"text", 'archived'::"text"])))
);


ALTER TABLE "public"."blog_posts" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."bot_personality_details" (
    "bot_id" "text" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "bot_name" "text",
    "bot_city" "text",
    "bot_country" "text",
    "bot_gender" "text",
    "bot_user_relation" "text"
);


ALTER TABLE "public"."bot_personality_details" OWNER TO "postgres";


COMMENT ON TABLE "public"."bot_personality_details" IS 'It stores bot_personality_details with bot name, bot id as primary key, bot gender, bot city, bot country';



CREATE TABLE IF NOT EXISTS "public"."categorization_progress" (
    "email" "text" NOT NULL,
    "bot_id" "text" NOT NULL,
    "last_processed_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


ALTER TABLE "public"."categorization_progress" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."chat_message_logs" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "user_id" "text" NOT NULL,
    "user_message" "text",
    "bot_response" "text",
    "timestamp" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."chat_message_logs" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."conversations" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "bot_id" character varying(255) NOT NULL,
    "user_email" character varying(255) NOT NULL,
    "username" character varying(255) DEFAULT 'User'::character varying,
    "user_message" "text" NOT NULL,
    "previous_conversation" "text",
    "image_url" "text",
    "image_base64" "text",
    "created_at" timestamp with time zone DEFAULT "now"(),
    "updated_at" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."conversations" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."emotion_contexts" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "conversation_id" "uuid",
    "emotion" character varying(255),
    "location" character varying(255),
    "action" character varying(255),
    "created_at" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."emotion_contexts" OWNER TO "postgres";


CREATE OR REPLACE VIEW "public"."conversation_details" AS
 SELECT "c"."id",
    "c"."bot_id",
    "c"."user_email",
    "c"."username",
    "c"."user_message",
    "c"."previous_conversation",
    "c"."image_url",
    "c"."image_base64",
    "c"."created_at",
    "c"."updated_at",
    "ec"."emotion",
    "ec"."location",
    "ec"."action"
   FROM ("public"."conversations" "c"
     LEFT JOIN "public"."emotion_contexts" "ec" ON (("c"."id" = "ec"."conversation_id")));


ALTER TABLE "public"."conversation_details" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."delta_category" (
    "id" "uuid" DEFAULT "extensions"."uuid_generate_v4"() NOT NULL,
    "email" "text" NOT NULL,
    "bot_id" "text" NOT NULL,
    "message1" "text" NOT NULL,
    "message2" "text" NOT NULL,
    "relation" "text" NOT NULL,
    "category" "text" NOT NULL,
    "output" "text",
    "timestamp" timestamp with time zone DEFAULT "now"() NOT NULL,
    "message1_id" "uuid",
    "message2_id" "uuid",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


ALTER TABLE "public"."delta_category" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."frontend_error_logs" (
    "message" "text" NOT NULL,
    "source" character varying NOT NULL,
    "line_number" integer,
    "column_number" integer,
    "stack_trace" "text",
    "browser" character varying,
    "url" character varying,
    "additional_context" "text",
    "timestamp" timestamp with time zone
);


ALTER TABLE "public"."frontend_error_logs" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."image_interpreter" (
    "id" integer NOT NULL,
    "bot_id" character varying(50) NOT NULL,
    "image_base64" "text" NOT NULL,
    "image_description" "text",
    "image_summary" "text",
    "final_response" "text",
    "created_at" timestamp with time zone DEFAULT "now"(),
    "updated_at" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."image_interpreter" OWNER TO "postgres";


CREATE SEQUENCE IF NOT EXISTS "public"."image_interpreter_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE "public"."image_interpreter_id_seq" OWNER TO "postgres";


ALTER SEQUENCE "public"."image_interpreter_id_seq" OWNED BY "public"."image_interpreter"."id";



CREATE TABLE IF NOT EXISTS "public"."last_cat_message" (
    "id" bigint NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "email" "text",
    "message_id" "text",
    "bot_id" "text",
    "morning_message" "text" DEFAULT ''::"text"
);


ALTER TABLE "public"."last_cat_message" OWNER TO "postgres";


ALTER TABLE "public"."last_cat_message" ALTER COLUMN "id" ADD GENERATED BY DEFAULT AS IDENTITY (
    SEQUENCE NAME "public"."last_cat_message_id_seq"
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);



CREATE TABLE IF NOT EXISTS "public"."log_messages_with_like_dislike" (
    "id" bigint NOT NULL,
    "user_email" "text" DEFAULT ''::"text",
    "bot_id" "text" DEFAULT ''::"text",
    "user_message" "text" DEFAULT ''::"text",
    "bot_response" "text" DEFAULT ''::"text",
    "feedback" "text" DEFAULT ''::"text",
    "last_5_messages" "text" DEFAULT ''::"text",
    "memory_retrieved" "text" DEFAULT ''::"text",
    "memory_extracted" "text" DEFAULT ''::"text",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


ALTER TABLE "public"."log_messages_with_like_dislike" OWNER TO "postgres";


ALTER TABLE "public"."log_messages_with_like_dislike" ALTER COLUMN "id" ADD GENERATED BY DEFAULT AS IDENTITY (
    SEQUENCE NAME "public"."log_messages_with_like_dislike_id_seq"
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);



CREATE TABLE IF NOT EXISTS "public"."message_paritition" (
    "id" bigint NOT NULL,
    "email" "text" DEFAULT ''::"text" NOT NULL,
    "user_message" "text" DEFAULT ''::"text",
    "bot_response" "text" DEFAULT ''::"text",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "bot_id" "text" DEFAULT ''::"text",
    "requested_time" "text" DEFAULT ''::"text",
    "platform" "text" DEFAULT ''::"text",
    "activity_name" "text" DEFAULT ''::"text",
    "timestamp" timestamp with time zone DEFAULT "now"()
)
PARTITION BY HASH ("email");


ALTER TABLE "public"."message_paritition" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."message_paritition_0" (
    "id" bigint NOT NULL,
    "email" "text" DEFAULT ''::"text" NOT NULL,
    "user_message" "text" DEFAULT ''::"text",
    "bot_response" "text" DEFAULT ''::"text",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "bot_id" "text" DEFAULT ''::"text",
    "requested_time" "text" DEFAULT ''::"text",
    "platform" "text" DEFAULT ''::"text",
    "activity_name" "text" DEFAULT ''::"text",
    "timestamp" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."message_paritition_0" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."message_paritition_1" (
    "id" bigint NOT NULL,
    "email" "text" DEFAULT ''::"text" NOT NULL,
    "user_message" "text" DEFAULT ''::"text",
    "bot_response" "text" DEFAULT ''::"text",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "bot_id" "text" DEFAULT ''::"text",
    "requested_time" "text" DEFAULT ''::"text",
    "platform" "text" DEFAULT ''::"text",
    "activity_name" "text" DEFAULT ''::"text",
    "timestamp" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."message_paritition_1" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."message_paritition_2" (
    "id" bigint NOT NULL,
    "email" "text" DEFAULT ''::"text" NOT NULL,
    "user_message" "text" DEFAULT ''::"text",
    "bot_response" "text" DEFAULT ''::"text",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "bot_id" "text" DEFAULT ''::"text",
    "requested_time" "text" DEFAULT ''::"text",
    "platform" "text" DEFAULT ''::"text",
    "activity_name" "text" DEFAULT ''::"text",
    "timestamp" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."message_paritition_2" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."message_paritition_3" (
    "id" bigint NOT NULL,
    "email" "text" DEFAULT ''::"text" NOT NULL,
    "user_message" "text" DEFAULT ''::"text",
    "bot_response" "text" DEFAULT ''::"text",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "bot_id" "text" DEFAULT ''::"text",
    "requested_time" "text" DEFAULT ''::"text",
    "platform" "text" DEFAULT ''::"text",
    "activity_name" "text" DEFAULT ''::"text",
    "timestamp" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."message_paritition_3" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."message_paritition_4" (
    "id" bigint NOT NULL,
    "email" "text" DEFAULT ''::"text" NOT NULL,
    "user_message" "text" DEFAULT ''::"text",
    "bot_response" "text" DEFAULT ''::"text",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "bot_id" "text" DEFAULT ''::"text",
    "requested_time" "text" DEFAULT ''::"text",
    "platform" "text" DEFAULT ''::"text",
    "activity_name" "text" DEFAULT ''::"text",
    "timestamp" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."message_paritition_4" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."message_paritition_5" (
    "id" bigint NOT NULL,
    "email" "text" DEFAULT ''::"text" NOT NULL,
    "user_message" "text" DEFAULT ''::"text",
    "bot_response" "text" DEFAULT ''::"text",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "bot_id" "text" DEFAULT ''::"text",
    "requested_time" "text" DEFAULT ''::"text",
    "platform" "text" DEFAULT ''::"text",
    "activity_name" "text" DEFAULT ''::"text",
    "timestamp" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."message_paritition_5" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."message_paritition_6" (
    "id" bigint NOT NULL,
    "email" "text" DEFAULT ''::"text" NOT NULL,
    "user_message" "text" DEFAULT ''::"text",
    "bot_response" "text" DEFAULT ''::"text",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "bot_id" "text" DEFAULT ''::"text",
    "requested_time" "text" DEFAULT ''::"text",
    "platform" "text" DEFAULT ''::"text",
    "activity_name" "text" DEFAULT ''::"text",
    "timestamp" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."message_paritition_6" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."message_paritition_7" (
    "id" bigint NOT NULL,
    "email" "text" DEFAULT ''::"text" NOT NULL,
    "user_message" "text" DEFAULT ''::"text",
    "bot_response" "text" DEFAULT ''::"text",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "bot_id" "text" DEFAULT ''::"text",
    "requested_time" "text" DEFAULT ''::"text",
    "platform" "text" DEFAULT ''::"text",
    "activity_name" "text" DEFAULT ''::"text",
    "timestamp" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."message_paritition_7" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."message_paritition_8" (
    "id" bigint NOT NULL,
    "email" "text" DEFAULT ''::"text" NOT NULL,
    "user_message" "text" DEFAULT ''::"text",
    "bot_response" "text" DEFAULT ''::"text",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "bot_id" "text" DEFAULT ''::"text",
    "requested_time" "text" DEFAULT ''::"text",
    "platform" "text" DEFAULT ''::"text",
    "activity_name" "text" DEFAULT ''::"text",
    "timestamp" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."message_paritition_8" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."message_paritition_9" (
    "id" bigint NOT NULL,
    "email" "text" DEFAULT ''::"text" NOT NULL,
    "user_message" "text" DEFAULT ''::"text",
    "bot_response" "text" DEFAULT ''::"text",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "bot_id" "text" DEFAULT ''::"text",
    "requested_time" "text" DEFAULT ''::"text",
    "platform" "text" DEFAULT ''::"text",
    "activity_name" "text" DEFAULT ''::"text",
    "timestamp" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."message_paritition_9" OWNER TO "postgres";


ALTER TABLE "public"."message_paritition" ALTER COLUMN "id" ADD GENERATED BY DEFAULT AS IDENTITY (
    SEQUENCE NAME "public"."message_paritition_id_seq"
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);



CREATE TABLE IF NOT EXISTS "public"."messages" (
    "id" bigint NOT NULL,
    "email" "text" DEFAULT ''::"text",
    "user_message" "text" DEFAULT ''::"text",
    "bot_response" "text" DEFAULT ''::"text",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "bot_id" "text" DEFAULT ''::"text",
    "requested_time" "text" DEFAULT ''::"text",
    "platform" "text"
);


ALTER TABLE "public"."messages" OWNER TO "postgres";


ALTER TABLE "public"."messages" ALTER COLUMN "id" ADD GENERATED BY DEFAULT AS IDENTITY (
    SEQUENCE NAME "public"."messages_id_seq"
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);



CREATE TABLE IF NOT EXISTS "public"."notes" (
    "id" bigint NOT NULL,
    "notes" "text" DEFAULT ''::"text",
    "email" "text",
    "bot_id" "text",
    "extracted_data" "text",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


ALTER TABLE "public"."notes" OWNER TO "postgres";


ALTER TABLE "public"."notes" ALTER COLUMN "id" ADD GENERATED BY DEFAULT AS IDENTITY (
    SEQUENCE NAME "public"."notes_id_seq"
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);



CREATE TABLE IF NOT EXISTS "public"."payment_transactions" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "stripe_session_id" "text" NOT NULL,
    "user_id" "uuid" NOT NULL,
    "stripe_customer_id" "text" NOT NULL,
    "price_id" "text" NOT NULL,
    "payment_amount" numeric(10,2),
    "processed_at" timestamp with time zone DEFAULT "now"(),
    "created_at" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."payment_transactions" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."persona_category" (
    "email" "text" NOT NULL,
    "bot_id" "text" NOT NULL,
    "category" "text",
    "memory" "text",
    "created_at" timestamp with time zone DEFAULT "now"(),
    "redundant" boolean DEFAULT false,
    "id" integer NOT NULL,
    "relation_id" "uuid",
    "embedding" "public"."vector"(768),
    "magnitude" real,
    "recency" smallint,
    "frequency" integer DEFAULT 1 NOT NULL,
    "rfm_score" real
);


ALTER TABLE "public"."persona_category" OWNER TO "postgres";


COMMENT ON COLUMN "public"."persona_category"."redundant" IS 'a column which captures if the corresponding rephrased message is duplicated or not';



CREATE TABLE IF NOT EXISTS "public"."persona_category_2" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "user_id" "text" NOT NULL,
    "memory_text" "text",
    "embedding" "public"."vector"(768),
    "created_at" timestamp with time zone DEFAULT "now"(),
    "last_used" timestamp with time zone DEFAULT "now"(),
    "frequency" integer DEFAULT 1,
    "magnitude" real,
    "rfm_score" real
);


ALTER TABLE "public"."persona_category_2" OWNER TO "postgres";


CREATE SEQUENCE IF NOT EXISTS "public"."persona_category_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE "public"."persona_category_id_seq" OWNER TO "postgres";


ALTER SEQUENCE "public"."persona_category_id_seq" OWNED BY "public"."persona_category"."id";



CREATE TABLE IF NOT EXISTS "public"."proactive_messages" (
    "id" bigint NOT NULL,
    "message" "text" DEFAULT ''::"text",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


ALTER TABLE "public"."proactive_messages" OWNER TO "postgres";


ALTER TABLE "public"."proactive_messages" ALTER COLUMN "id" ADD GENERATED BY DEFAULT AS IDENTITY (
    SEQUENCE NAME "public"."proactive_messages_id_seq"
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);



CREATE OR REPLACE VIEW "public"."public_blog_posts" AS
 SELECT "blog_posts"."id",
    "blog_posts"."title",
    "blog_posts"."slug",
    "blog_posts"."content",
    "blog_posts"."excerpt",
    "blog_posts"."tags",
    "blog_posts"."publish_date",
    "blog_posts"."author_email",
    "blog_posts"."featured_image_url",
    "blog_posts"."additional_images",
    "blog_posts"."view_count",
    "blog_posts"."created_at",
    "blog_posts"."updated_at"
   FROM "public"."blog_posts"
  WHERE ("blog_posts"."status" = 'published'::"text")
  ORDER BY "blog_posts"."created_at" DESC;


ALTER TABLE "public"."public_blog_posts" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."retrieve_memory_data" (
    "id" bigint NOT NULL,
    "previous_conversations" "text",
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "extracted_data" "text",
    "email" "text",
    "bot_id" "text"
);


ALTER TABLE "public"."retrieve_memory_data" OWNER TO "postgres";


ALTER TABLE "public"."retrieve_memory_data" ALTER COLUMN "id" ADD GENERATED BY DEFAULT AS IDENTITY (
    SEQUENCE NAME "public"."retrieve_memory_data_id_seq"
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);



CREATE TABLE IF NOT EXISTS "public"."scheduler" (
    "id" bigint NOT NULL,
    "email" "text",
    "bot_id" "text",
    "message" "text",
    "scheduled_time" timestamp with time zone,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "user_timezone_offset" "text" DEFAULT ''::"text"
);


ALTER TABLE "public"."scheduler" OWNER TO "postgres";


ALTER TABLE "public"."scheduler" ALTER COLUMN "id" ADD GENERATED BY DEFAULT AS IDENTITY (
    SEQUENCE NAME "public"."scheduler_id_seq"
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);



CREATE TABLE IF NOT EXISTS "public"."summary" (
    "email" character varying(255) NOT NULL,
    "bot_id" "text" NOT NULL,
    "generated_summary" "text" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "summary_date" "date"
);


ALTER TABLE "public"."summary" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."todos" (
    "id" integer NOT NULL,
    "name" "text" NOT NULL
);


ALTER TABLE "public"."todos" OWNER TO "postgres";


CREATE SEQUENCE IF NOT EXISTS "public"."todos_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE "public"."todos_id_seq" OWNER TO "postgres";


ALTER SEQUENCE "public"."todos_id_seq" OWNED BY "public"."todos"."id";



CREATE TABLE IF NOT EXISTS "public"."user_details" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "email" "text",
    "user_id" "uuid",
    "name" "text" NOT NULL,
    "gender" "text",
    "city" "text",
    "created_at" timestamp without time zone DEFAULT "now"(),
    "auth_provider" "text",
    "subscription_status" "text" DEFAULT 'Free trial'::"text" NOT NULL,
    "subscription_duration" "text",
    "payment_date" "date",
    "payment_amount" numeric(10,2),
    "subscription_expires_at" "date",
    "stripe_customer_id" "text",
    "current_plan" "text",
    "role" "text" DEFAULT 'user'::"text",
    "timezone" "text" DEFAULT 'UTC'::"text",
    CONSTRAINT "user_details_gender_check" CHECK (("gender" = ANY (ARRAY['Male'::"text", 'Female'::"text", 'Other'::"text"]))),
    CONSTRAINT "user_details_role_check" CHECK (("role" = ANY (ARRAY['user'::"text", 'admin'::"text"])))
);


ALTER TABLE "public"."user_details" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."user_xp" (
    "id" bigint NOT NULL,
    "email" "text" NOT NULL,
    "bot_id" "text",
    "xp_score" integer DEFAULT 0,
    "coins" integer DEFAULT 0,
    "magnitude" double precision DEFAULT 0,
    "updated_at" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."user_xp" OWNER TO "postgres";


CREATE SEQUENCE IF NOT EXISTS "public"."user_xp_id_seq"
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE "public"."user_xp_id_seq" OWNER TO "postgres";


ALTER SEQUENCE "public"."user_xp_id_seq" OWNED BY "public"."user_xp"."id";



ALTER TABLE ONLY "public"."message_paritition" ATTACH PARTITION "public"."message_paritition_0" FOR VALUES WITH (modulus 10, remainder 0);



ALTER TABLE ONLY "public"."message_paritition" ATTACH PARTITION "public"."message_paritition_1" FOR VALUES WITH (modulus 10, remainder 1);



ALTER TABLE ONLY "public"."message_paritition" ATTACH PARTITION "public"."message_paritition_2" FOR VALUES WITH (modulus 10, remainder 2);



ALTER TABLE ONLY "public"."message_paritition" ATTACH PARTITION "public"."message_paritition_3" FOR VALUES WITH (modulus 10, remainder 3);



ALTER TABLE ONLY "public"."message_paritition" ATTACH PARTITION "public"."message_paritition_4" FOR VALUES WITH (modulus 10, remainder 4);



ALTER TABLE ONLY "public"."message_paritition" ATTACH PARTITION "public"."message_paritition_5" FOR VALUES WITH (modulus 10, remainder 5);



ALTER TABLE ONLY "public"."message_paritition" ATTACH PARTITION "public"."message_paritition_6" FOR VALUES WITH (modulus 10, remainder 6);



ALTER TABLE ONLY "public"."message_paritition" ATTACH PARTITION "public"."message_paritition_7" FOR VALUES WITH (modulus 10, remainder 7);



ALTER TABLE ONLY "public"."message_paritition" ATTACH PARTITION "public"."message_paritition_8" FOR VALUES WITH (modulus 10, remainder 8);



ALTER TABLE ONLY "public"."message_paritition" ATTACH PARTITION "public"."message_paritition_9" FOR VALUES WITH (modulus 10, remainder 9);



ALTER TABLE ONLY "public"."image_interpreter" ALTER COLUMN "id" SET DEFAULT "nextval"('"public"."image_interpreter_id_seq"'::"regclass");



ALTER TABLE ONLY "public"."persona_category" ALTER COLUMN "id" SET DEFAULT "nextval"('"public"."persona_category_id_seq"'::"regclass");



ALTER TABLE ONLY "public"."todos" ALTER COLUMN "id" SET DEFAULT "nextval"('"public"."todos_id_seq"'::"regclass");



ALTER TABLE ONLY "public"."user_xp" ALTER COLUMN "id" SET DEFAULT "nextval"('"public"."user_xp_id_seq"'::"regclass");



ALTER TABLE ONLY "public"."blog_posts"
    ADD CONSTRAINT "blog_posts_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."blog_posts"
    ADD CONSTRAINT "blog_posts_slug_key" UNIQUE ("slug");



ALTER TABLE ONLY "public"."bot_personality_details"
    ADD CONSTRAINT "bot_personality_details_pkey" PRIMARY KEY ("bot_id");



ALTER TABLE ONLY "public"."categorization_progress"
    ADD CONSTRAINT "categorization_progress_pkey" PRIMARY KEY ("email", "bot_id");



ALTER TABLE ONLY "public"."chat_message_logs"
    ADD CONSTRAINT "chat_message_logs_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."conversations"
    ADD CONSTRAINT "conversations_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."delta_category"
    ADD CONSTRAINT "delta_category_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."emotion_contexts"
    ADD CONSTRAINT "emotion_contexts_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."image_interpreter"
    ADD CONSTRAINT "image_interpreter_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."last_cat_message"
    ADD CONSTRAINT "lcm_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."log_messages_with_like_dislike"
    ADD CONSTRAINT "log_messages_with_like_dislike_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."message_paritition"
    ADD CONSTRAINT "message_partition_pkey" PRIMARY KEY ("id", "email");



ALTER TABLE ONLY "public"."message_paritition_0"
    ADD CONSTRAINT "message_paritition_0_pkey" PRIMARY KEY ("id", "email");



ALTER TABLE ONLY "public"."message_paritition_1"
    ADD CONSTRAINT "message_paritition_1_pkey" PRIMARY KEY ("id", "email");



ALTER TABLE ONLY "public"."message_paritition_2"
    ADD CONSTRAINT "message_paritition_2_pkey" PRIMARY KEY ("id", "email");



ALTER TABLE ONLY "public"."message_paritition_3"
    ADD CONSTRAINT "message_paritition_3_pkey" PRIMARY KEY ("id", "email");



ALTER TABLE ONLY "public"."message_paritition_4"
    ADD CONSTRAINT "message_paritition_4_pkey" PRIMARY KEY ("id", "email");



ALTER TABLE ONLY "public"."message_paritition_5"
    ADD CONSTRAINT "message_paritition_5_pkey" PRIMARY KEY ("id", "email");



ALTER TABLE ONLY "public"."message_paritition_6"
    ADD CONSTRAINT "message_paritition_6_pkey" PRIMARY KEY ("id", "email");



ALTER TABLE ONLY "public"."message_paritition_7"
    ADD CONSTRAINT "message_paritition_7_pkey" PRIMARY KEY ("id", "email");



ALTER TABLE ONLY "public"."message_paritition_8"
    ADD CONSTRAINT "message_paritition_8_pkey" PRIMARY KEY ("id", "email");



ALTER TABLE ONLY "public"."message_paritition_9"
    ADD CONSTRAINT "message_paritition_9_pkey" PRIMARY KEY ("id", "email");



ALTER TABLE ONLY "public"."messages"
    ADD CONSTRAINT "messages_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."notes"
    ADD CONSTRAINT "notes_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."payment_transactions"
    ADD CONSTRAINT "payment_transactions_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."payment_transactions"
    ADD CONSTRAINT "payment_transactions_stripe_session_id_key" UNIQUE ("stripe_session_id");



ALTER TABLE ONLY "public"."persona_category_2"
    ADD CONSTRAINT "persona_category_2_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."persona_category"
    ADD CONSTRAINT "persona_category_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."proactive_messages"
    ADD CONSTRAINT "proactive_messages_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."retrieve_memory_data"
    ADD CONSTRAINT "retrieve_memory_data_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."todos"
    ADD CONSTRAINT "todos_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."user_details"
    ADD CONSTRAINT "user_details_email_key" UNIQUE ("email");



ALTER TABLE ONLY "public"."user_details"
    ADD CONSTRAINT "user_details_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."user_details"
    ADD CONSTRAINT "user_details_stripe_customer_id_key" UNIQUE ("stripe_customer_id");



ALTER TABLE ONLY "public"."user_xp"
    ADD CONSTRAINT "user_xp_email_bot_id_unique" UNIQUE ("email", "bot_id");



ALTER TABLE ONLY "public"."user_xp"
    ADD CONSTRAINT "user_xp_pkey" PRIMARY KEY ("id");



CREATE INDEX "idx_blog_posts_author_email" ON "public"."blog_posts" USING "btree" ("author_email");



CREATE INDEX "idx_blog_posts_created_at" ON "public"."blog_posts" USING "btree" ("created_at");



CREATE INDEX "idx_blog_posts_publish_date" ON "public"."blog_posts" USING "btree" ("publish_date");



CREATE INDEX "idx_blog_posts_slug" ON "public"."blog_posts" USING "btree" ("slug");



CREATE INDEX "idx_blog_posts_status" ON "public"."blog_posts" USING "btree" ("status");



CREATE INDEX "idx_blog_posts_tags" ON "public"."blog_posts" USING "gin" ("tags");



CREATE INDEX "idx_conversations_bot_id" ON "public"."conversations" USING "btree" ("bot_id");



CREATE INDEX "idx_conversations_created_at" ON "public"."conversations" USING "btree" ("created_at");



CREATE INDEX "idx_conversations_user_email" ON "public"."conversations" USING "btree" ("user_email");



CREATE INDEX "idx_emotion_contexts_conversation_id" ON "public"."emotion_contexts" USING "btree" ("conversation_id");



CREATE INDEX "idx_image_interpreter_bot_id" ON "public"."image_interpreter" USING "btree" ("bot_id");



CREATE INDEX "idx_image_interpreter_created_at" ON "public"."image_interpreter" USING "btree" ("created_at");



CREATE INDEX "idx_payment_transactions_session_id" ON "public"."payment_transactions" USING "btree" ("stripe_session_id");



CREATE INDEX "idx_user_details_stripe_customer_id" ON "public"."user_details" USING "btree" ("stripe_customer_id");



CREATE INDEX "idx_user_details_timezone" ON "public"."user_details" USING "btree" ("timezone");



CREATE INDEX "persona_category_2_embedding_hnsw_idx" ON "public"."persona_category_2" USING "hnsw" ("embedding" "public"."vector_cosine_ops") WITH ("m"='16', "ef_construction"='64');



CREATE INDEX "user_details_role_idx" ON "public"."user_details" USING "btree" ("role");



ALTER INDEX "public"."message_partition_pkey" ATTACH PARTITION "public"."message_paritition_0_pkey";



ALTER INDEX "public"."message_partition_pkey" ATTACH PARTITION "public"."message_paritition_1_pkey";



ALTER INDEX "public"."message_partition_pkey" ATTACH PARTITION "public"."message_paritition_2_pkey";



ALTER INDEX "public"."message_partition_pkey" ATTACH PARTITION "public"."message_paritition_3_pkey";



ALTER INDEX "public"."message_partition_pkey" ATTACH PARTITION "public"."message_paritition_4_pkey";



ALTER INDEX "public"."message_partition_pkey" ATTACH PARTITION "public"."message_paritition_5_pkey";



ALTER INDEX "public"."message_partition_pkey" ATTACH PARTITION "public"."message_paritition_6_pkey";



ALTER INDEX "public"."message_partition_pkey" ATTACH PARTITION "public"."message_paritition_7_pkey";



ALTER INDEX "public"."message_partition_pkey" ATTACH PARTITION "public"."message_paritition_8_pkey";



ALTER INDEX "public"."message_partition_pkey" ATTACH PARTITION "public"."message_paritition_9_pkey";



CREATE OR REPLACE TRIGGER "auto_generate_slug_trigger" BEFORE INSERT ON "public"."blog_posts" FOR EACH ROW EXECUTE FUNCTION "public"."auto_generate_slug"();



CREATE OR REPLACE TRIGGER "trigger_set_blog_post_slug" BEFORE INSERT OR UPDATE ON "public"."blog_posts" FOR EACH ROW EXECUTE FUNCTION "public"."set_blog_post_slug"();



CREATE OR REPLACE TRIGGER "trigger_update_blog_posts_updated_at" BEFORE UPDATE ON "public"."blog_posts" FOR EACH ROW EXECUTE FUNCTION "public"."update_updated_at_column"();



CREATE OR REPLACE TRIGGER "update_blog_posts_updated_at" BEFORE UPDATE ON "public"."blog_posts" FOR EACH ROW EXECUTE FUNCTION "public"."update_updated_at_column"();



CREATE OR REPLACE TRIGGER "update_conversations_updated_at" BEFORE UPDATE ON "public"."conversations" FOR EACH ROW EXECUTE FUNCTION "public"."update_updated_at_column"();



CREATE OR REPLACE TRIGGER "update_image_interpreter_updated_at" BEFORE UPDATE ON "public"."image_interpreter" FOR EACH ROW EXECUTE FUNCTION "public"."update_updated_at_column"();



ALTER TABLE ONLY "public"."emotion_contexts"
    ADD CONSTRAINT "emotion_contexts_conversation_id_fkey" FOREIGN KEY ("conversation_id") REFERENCES "public"."conversations"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."payment_transactions"
    ADD CONSTRAINT "payment_transactions_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "public"."user_details"("id");



ALTER TABLE ONLY "public"."user_details"
    ADD CONSTRAINT "user_details_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "auth"."users"("id");



CREATE POLICY "Allow error logging" ON "public"."frontend_error_logs" FOR INSERT WITH CHECK (true);



CREATE POLICY "Anyone can read published posts" ON "public"."blog_posts" FOR SELECT USING (("status" = 'published'::"text"));



CREATE POLICY "Authors can delete their own posts" ON "public"."blog_posts" FOR DELETE USING ((("auth"."jwt"() ->> 'email'::"text") = "author_email"));



CREATE POLICY "Authors can insert their own posts" ON "public"."blog_posts" FOR INSERT WITH CHECK ((("auth"."jwt"() ->> 'email'::"text") = "author_email"));



CREATE POLICY "Authors can read their own posts" ON "public"."blog_posts" FOR SELECT USING ((("auth"."jwt"() ->> 'email'::"text") = "author_email"));



CREATE POLICY "Authors can update their own posts" ON "public"."blog_posts" FOR UPDATE USING ((("auth"."jwt"() ->> 'email'::"text") = "author_email")) WITH CHECK ((("auth"."jwt"() ->> 'email'::"text") = "author_email"));



CREATE POLICY "Service role can do everything" ON "public"."blog_posts" USING (("auth"."role"() = 'service_role'::"text"));



CREATE POLICY "Users can Insert chats." ON "public"."message_paritition" FOR INSERT TO "authenticated" WITH CHECK ((( SELECT "auth"."email"() AS "email") = "email"));



CREATE POLICY "Users can Insert." ON "public"."last_cat_message" FOR INSERT TO "authenticated" WITH CHECK ((( SELECT "auth"."email"() AS "email") = "email"));



CREATE POLICY "Users can Select chats." ON "public"."message_paritition" FOR SELECT TO "authenticated" USING (("email" = "auth"."email"()));



CREATE POLICY "Users can delete their own posts" ON "public"."blog_posts" FOR DELETE USING ((("auth"."jwt"() ->> 'email'::"text") = "author_email"));



CREATE POLICY "Users can insert their own posts" ON "public"."blog_posts" FOR INSERT WITH CHECK ((("auth"."jwt"() ->> 'email'::"text") = "author_email"));



CREATE POLICY "Users can read published posts" ON "public"."blog_posts" FOR SELECT USING (("status" = 'published'::"text"));



CREATE POLICY "Users can read their own posts" ON "public"."blog_posts" FOR SELECT USING ((("auth"."jwt"() ->> 'email'::"text") = "author_email"));



CREATE POLICY "Users can update their own posts" ON "public"."blog_posts" FOR UPDATE USING ((("auth"."jwt"() ->> 'email'::"text") = "author_email"));



CREATE POLICY "allow_anon_insert" ON "public"."last_cat_message" FOR INSERT TO "anon" WITH CHECK (true);



CREATE POLICY "allow_anon_insert" ON "public"."message_paritition" FOR INSERT TO "anon" WITH CHECK (true);



CREATE POLICY "allow_anon_insert" ON "public"."user_details" FOR INSERT TO "anon" WITH CHECK (true);



CREATE POLICY "allow_anon_select" ON "public"."last_cat_message" FOR SELECT TO "anon" USING (true);



CREATE POLICY "allow_anon_select" ON "public"."message_paritition" FOR SELECT TO "anon" USING (true);



CREATE POLICY "allow_anon_select" ON "public"."user_details" FOR SELECT TO "anon" USING (true);



CREATE POLICY "allow_anon_update" ON "public"."last_cat_message" FOR UPDATE TO "anon" WITH CHECK (true);



CREATE POLICY "allow_anon_update" ON "public"."message_paritition" FOR UPDATE TO "anon" WITH CHECK (true);



CREATE POLICY "allow_anon_update" ON "public"."user_details" FOR UPDATE TO "anon" WITH CHECK (true);



ALTER TABLE "public"."message_paritition" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."message_paritition_9" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."messages" ENABLE ROW LEVEL SECURITY;


GRANT USAGE ON SCHEMA "public" TO "postgres";
GRANT USAGE ON SCHEMA "public" TO "anon";
GRANT USAGE ON SCHEMA "public" TO "authenticated";
GRANT USAGE ON SCHEMA "public" TO "service_role";



GRANT ALL ON FUNCTION "public"."auto_generate_slug"() TO "anon";
GRANT ALL ON FUNCTION "public"."auto_generate_slug"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."auto_generate_slug"() TO "service_role";



GRANT ALL ON FUNCTION "public"."cleanup_old_drafts"("days_old" integer) TO "anon";
GRANT ALL ON FUNCTION "public"."cleanup_old_drafts"("days_old" integer) TO "authenticated";
GRANT ALL ON FUNCTION "public"."cleanup_old_drafts"("days_old" integer) TO "service_role";



GRANT ALL ON FUNCTION "public"."generate_slug"("title" "text") TO "anon";
GRANT ALL ON FUNCTION "public"."generate_slug"("title" "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."generate_slug"("title" "text") TO "service_role";



GRANT ALL ON FUNCTION "public"."get_blog_stats"() TO "anon";
GRANT ALL ON FUNCTION "public"."get_blog_stats"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."get_blog_stats"() TO "service_role";



GRANT ALL ON FUNCTION "public"."increment_view_count"("post_id" "uuid") TO "anon";
GRANT ALL ON FUNCTION "public"."increment_view_count"("post_id" "uuid") TO "authenticated";
GRANT ALL ON FUNCTION "public"."increment_view_count"("post_id" "uuid") TO "service_role";



GRANT ALL ON FUNCTION "public"."search_blog_posts"("search_term" "text", "tag_filter" "text", "author_filter" "text", "status_filter" "text", "page_num" integer, "page_size" integer) TO "anon";
GRANT ALL ON FUNCTION "public"."search_blog_posts"("search_term" "text", "tag_filter" "text", "author_filter" "text", "status_filter" "text", "page_num" integer, "page_size" integer) TO "authenticated";
GRANT ALL ON FUNCTION "public"."search_blog_posts"("search_term" "text", "tag_filter" "text", "author_filter" "text", "status_filter" "text", "page_num" integer, "page_size" integer) TO "service_role";



GRANT ALL ON FUNCTION "public"."set_blog_post_slug"() TO "anon";
GRANT ALL ON FUNCTION "public"."set_blog_post_slug"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."set_blog_post_slug"() TO "service_role";



GRANT ALL ON FUNCTION "public"."update_updated_at_column"() TO "anon";
GRANT ALL ON FUNCTION "public"."update_updated_at_column"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."update_updated_at_column"() TO "service_role";



GRANT ALL ON TABLE "public"."blog_posts" TO "anon";
GRANT ALL ON TABLE "public"."blog_posts" TO "authenticated";
GRANT ALL ON TABLE "public"."blog_posts" TO "service_role";



GRANT ALL ON TABLE "public"."bot_personality_details" TO "anon";
GRANT ALL ON TABLE "public"."bot_personality_details" TO "authenticated";
GRANT ALL ON TABLE "public"."bot_personality_details" TO "service_role";



GRANT ALL ON TABLE "public"."categorization_progress" TO "anon";
GRANT ALL ON TABLE "public"."categorization_progress" TO "authenticated";
GRANT ALL ON TABLE "public"."categorization_progress" TO "service_role";



GRANT ALL ON TABLE "public"."chat_message_logs" TO "anon";
GRANT ALL ON TABLE "public"."chat_message_logs" TO "authenticated";
GRANT ALL ON TABLE "public"."chat_message_logs" TO "service_role";



GRANT ALL ON TABLE "public"."conversations" TO "anon";
GRANT ALL ON TABLE "public"."conversations" TO "authenticated";
GRANT ALL ON TABLE "public"."conversations" TO "service_role";



GRANT ALL ON TABLE "public"."emotion_contexts" TO "anon";
GRANT ALL ON TABLE "public"."emotion_contexts" TO "authenticated";
GRANT ALL ON TABLE "public"."emotion_contexts" TO "service_role";



GRANT ALL ON TABLE "public"."conversation_details" TO "anon";
GRANT ALL ON TABLE "public"."conversation_details" TO "authenticated";
GRANT ALL ON TABLE "public"."conversation_details" TO "service_role";



GRANT ALL ON TABLE "public"."delta_category" TO "anon";
GRANT ALL ON TABLE "public"."delta_category" TO "authenticated";
GRANT ALL ON TABLE "public"."delta_category" TO "service_role";



GRANT ALL ON TABLE "public"."frontend_error_logs" TO "anon";
GRANT ALL ON TABLE "public"."frontend_error_logs" TO "authenticated";
GRANT ALL ON TABLE "public"."frontend_error_logs" TO "service_role";



GRANT ALL ON TABLE "public"."image_interpreter" TO "anon";
GRANT ALL ON TABLE "public"."image_interpreter" TO "authenticated";
GRANT ALL ON TABLE "public"."image_interpreter" TO "service_role";



GRANT ALL ON SEQUENCE "public"."image_interpreter_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."image_interpreter_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."image_interpreter_id_seq" TO "service_role";



GRANT ALL ON TABLE "public"."last_cat_message" TO "anon";
GRANT ALL ON TABLE "public"."last_cat_message" TO "authenticated";
GRANT ALL ON TABLE "public"."last_cat_message" TO "service_role";



GRANT ALL ON SEQUENCE "public"."last_cat_message_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."last_cat_message_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."last_cat_message_id_seq" TO "service_role";



GRANT ALL ON TABLE "public"."log_messages_with_like_dislike" TO "anon";
GRANT ALL ON TABLE "public"."log_messages_with_like_dislike" TO "authenticated";
GRANT ALL ON TABLE "public"."log_messages_with_like_dislike" TO "service_role";



GRANT ALL ON SEQUENCE "public"."log_messages_with_like_dislike_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."log_messages_with_like_dislike_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."log_messages_with_like_dislike_id_seq" TO "service_role";



GRANT ALL ON TABLE "public"."message_paritition" TO "anon";
GRANT ALL ON TABLE "public"."message_paritition" TO "authenticated";
GRANT ALL ON TABLE "public"."message_paritition" TO "service_role";



GRANT ALL ON TABLE "public"."message_paritition_0" TO "anon";
GRANT ALL ON TABLE "public"."message_paritition_0" TO "authenticated";
GRANT ALL ON TABLE "public"."message_paritition_0" TO "service_role";



GRANT ALL ON TABLE "public"."message_paritition_1" TO "anon";
GRANT ALL ON TABLE "public"."message_paritition_1" TO "authenticated";
GRANT ALL ON TABLE "public"."message_paritition_1" TO "service_role";



GRANT ALL ON TABLE "public"."message_paritition_2" TO "anon";
GRANT ALL ON TABLE "public"."message_paritition_2" TO "authenticated";
GRANT ALL ON TABLE "public"."message_paritition_2" TO "service_role";



GRANT ALL ON TABLE "public"."message_paritition_3" TO "anon";
GRANT ALL ON TABLE "public"."message_paritition_3" TO "authenticated";
GRANT ALL ON TABLE "public"."message_paritition_3" TO "service_role";



GRANT ALL ON TABLE "public"."message_paritition_4" TO "anon";
GRANT ALL ON TABLE "public"."message_paritition_4" TO "authenticated";
GRANT ALL ON TABLE "public"."message_paritition_4" TO "service_role";



GRANT ALL ON TABLE "public"."message_paritition_5" TO "anon";
GRANT ALL ON TABLE "public"."message_paritition_5" TO "authenticated";
GRANT ALL ON TABLE "public"."message_paritition_5" TO "service_role";



GRANT ALL ON TABLE "public"."message_paritition_6" TO "anon";
GRANT ALL ON TABLE "public"."message_paritition_6" TO "authenticated";
GRANT ALL ON TABLE "public"."message_paritition_6" TO "service_role";



GRANT ALL ON TABLE "public"."message_paritition_7" TO "anon";
GRANT ALL ON TABLE "public"."message_paritition_7" TO "authenticated";
GRANT ALL ON TABLE "public"."message_paritition_7" TO "service_role";



GRANT ALL ON TABLE "public"."message_paritition_8" TO "anon";
GRANT ALL ON TABLE "public"."message_paritition_8" TO "authenticated";
GRANT ALL ON TABLE "public"."message_paritition_8" TO "service_role";



GRANT ALL ON TABLE "public"."message_paritition_9" TO "anon";
GRANT ALL ON TABLE "public"."message_paritition_9" TO "authenticated";
GRANT ALL ON TABLE "public"."message_paritition_9" TO "service_role";



GRANT ALL ON SEQUENCE "public"."message_paritition_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."message_paritition_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."message_paritition_id_seq" TO "service_role";



GRANT ALL ON TABLE "public"."messages" TO "anon";
GRANT ALL ON TABLE "public"."messages" TO "authenticated";
GRANT ALL ON TABLE "public"."messages" TO "service_role";



GRANT ALL ON SEQUENCE "public"."messages_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."messages_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."messages_id_seq" TO "service_role";



GRANT ALL ON TABLE "public"."notes" TO "anon";
GRANT ALL ON TABLE "public"."notes" TO "authenticated";
GRANT ALL ON TABLE "public"."notes" TO "service_role";



GRANT ALL ON SEQUENCE "public"."notes_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."notes_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."notes_id_seq" TO "service_role";



GRANT ALL ON TABLE "public"."payment_transactions" TO "anon";
GRANT ALL ON TABLE "public"."payment_transactions" TO "authenticated";
GRANT ALL ON TABLE "public"."payment_transactions" TO "service_role";



GRANT ALL ON TABLE "public"."persona_category" TO "anon";
GRANT ALL ON TABLE "public"."persona_category" TO "authenticated";
GRANT ALL ON TABLE "public"."persona_category" TO "service_role";



GRANT ALL ON TABLE "public"."persona_category_2" TO "anon";
GRANT ALL ON TABLE "public"."persona_category_2" TO "authenticated";
GRANT ALL ON TABLE "public"."persona_category_2" TO "service_role";



GRANT ALL ON SEQUENCE "public"."persona_category_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."persona_category_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."persona_category_id_seq" TO "service_role";



GRANT ALL ON TABLE "public"."proactive_messages" TO "anon";
GRANT ALL ON TABLE "public"."proactive_messages" TO "authenticated";
GRANT ALL ON TABLE "public"."proactive_messages" TO "service_role";



GRANT ALL ON SEQUENCE "public"."proactive_messages_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."proactive_messages_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."proactive_messages_id_seq" TO "service_role";



GRANT ALL ON TABLE "public"."public_blog_posts" TO "anon";
GRANT ALL ON TABLE "public"."public_blog_posts" TO "authenticated";
GRANT ALL ON TABLE "public"."public_blog_posts" TO "service_role";



GRANT ALL ON TABLE "public"."retrieve_memory_data" TO "anon";
GRANT ALL ON TABLE "public"."retrieve_memory_data" TO "authenticated";
GRANT ALL ON TABLE "public"."retrieve_memory_data" TO "service_role";



GRANT ALL ON SEQUENCE "public"."retrieve_memory_data_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."retrieve_memory_data_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."retrieve_memory_data_id_seq" TO "service_role";



GRANT ALL ON TABLE "public"."scheduler" TO "anon";
GRANT ALL ON TABLE "public"."scheduler" TO "authenticated";
GRANT ALL ON TABLE "public"."scheduler" TO "service_role";



GRANT ALL ON SEQUENCE "public"."scheduler_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."scheduler_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."scheduler_id_seq" TO "service_role";



GRANT ALL ON TABLE "public"."summary" TO "anon";
GRANT ALL ON TABLE "public"."summary" TO "authenticated";
GRANT ALL ON TABLE "public"."summary" TO "service_role";



GRANT ALL ON TABLE "public"."todos" TO "anon";
GRANT ALL ON TABLE "public"."todos" TO "authenticated";
GRANT ALL ON TABLE "public"."todos" TO "service_role";



GRANT ALL ON SEQUENCE "public"."todos_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."todos_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."todos_id_seq" TO "service_role";



GRANT ALL ON TABLE "public"."user_details" TO "anon";
GRANT ALL ON TABLE "public"."user_details" TO "authenticated";
GRANT ALL ON TABLE "public"."user_details" TO "service_role";



GRANT ALL ON TABLE "public"."user_xp" TO "anon";
GRANT ALL ON TABLE "public"."user_xp" TO "authenticated";
GRANT ALL ON TABLE "public"."user_xp" TO "service_role";



GRANT ALL ON SEQUENCE "public"."user_xp_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."user_xp_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."user_xp_id_seq" TO "service_role";



ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES  TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES  TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES  TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES  TO "service_role";






ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS  TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS  TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS  TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS  TO "service_role";






ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES  TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES  TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES  TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES  TO "service_role";






RESET ALL;
