from fastapi import FastAPI, HTTPException, APIRouter, UploadFile, File
from azure.search.documents import SearchClient
from openai import AzureOpenAI
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)
from api.doc_extractors import *
from api.chunking_helpers import *
import os
import logging
from azure.storage.blob import BlobServiceClient
from azure.search.documents.indexes import SearchIndexerClient
from azure.core.credentials import AzureKeyCredential

# Azure Exceptions
from azure.core.exceptions import (
    AzureError,
    ResourceNotFoundError,
)

logger = logging.getLogger(__name__)
from config import *
from azure.search.documents.indexes import SearchIndexerClient
from azure.core.credentials import AzureKeyCredential

search_index_client = SearchIndexClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
)

search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
)

blob_service_client = BlobServiceClient.from_connection_string(
    AZURE_STORAGE_CONNECTION_STRING
)

embedding_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
)

router2 = APIRouter(tags=["Chunking and Embedding"])


@router2.post("/documents/chunk/{document_id}")
async def chunk_document(document_id: str):

    logger.info(
        "Chunking started | document_id=%s",
        document_id,
    )

    if not document_id:
        raise HTTPException(
            status_code=400,
            detail="Document ID is required.",
        )

    try:

        blob_client = blob_service_client.get_container_client(
            AZURE_STORAGE_CONTAINER4
        ).get_blob_client(f"{document_id}.txt")

        # Fetch processed text
        try:

            text = blob_client.download_blob().readall().decode("utf-8")

        except ResourceNotFoundError:

            raise HTTPException(
                status_code=404,
                detail="Processed document not found.",
            )

        except AzureError:

            logger.exception(
                "Blob retrieval failed | document_id=%s",
                document_id,
            )

            raise HTTPException(
                status_code=502,
                detail="Unable to retrieve processed document.",
            )

        text = text.strip()

        if not text:
            raise HTTPException(
                status_code=422,
                detail="Processed document contains no usable text.",
            )

        # Create chunks
        chunks = create_chunks(
            text=text,
            document_id=document_id,
            source_file=f"{document_id}.txt",
        )

        logger.info(
            "Chunking completed | document_id=%s | chunks=%d",
            document_id,
            len(chunks),
        )

        return {
            "status": "completed",
            "document_id": document_id,
            "chunk_count": len(chunks),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "strategy": "paragraph_based",
            "chunks": chunks,
        }

    except HTTPException:
        raise

    except Exception:

        logger.exception(
            "Chunking failed | document_id=%s",
            document_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Document chunking failed.",
        )


@router2.post("/documents/embed/{document_id}")
async def embed_document(document_id: str):

    logger.info(
        "Embedding started | document_id=%s",
        document_id,
    )

    if not document_id:
        raise HTTPException(
            status_code=400,
            detail="Document ID is required.",
        )

    try:

        # get processed text from container

        container_client = blob_service_client.get_container_client(
            AZURE_STORAGE_CONTAINER4
        )

        blob_client = container_client.get_blob_client(f"{document_id}.txt")

        try:

            file_content = blob_client.download_blob().readall()

        except ResourceNotFoundError:

            logger.warning(
                "Processed text not found | " "document_id=%s",
                document_id,
            )

            raise HTTPException(
                status_code=404,
                detail="Processed document not found.",
            )

        except AzureError:

            logger.exception(
                "Blob retrieval failed | " "document_id=%s",
                document_id,
            )

            raise HTTPException(
                status_code=502,
                detail="Failed to retrieve processed document.",
            )

        # Validate processed text

        text = file_content.decode("utf-8").strip()

        if not text:

            logger.warning(
                "Empty processed document | " "document_id=%s",
                document_id,
            )

            raise HTTPException(
                status_code=422,
                detail="Processed document contains no usable text.",
            )

        # create chunks

        chunks = create_chunks(
            text=text,
            document_id=document_id,
            source_file=f"{document_id}.txt",
        )

        if not chunks:

            raise HTTPException(
                status_code=422,
                detail="No chunks were created.",
            )

        logger.info(
            "Chunks created | document_id=%s | count=%d",
            document_id,
            len(chunks),
        )

        # Generate embeddings

        search_documents = []
        failed_chunks = []

        for chunk in chunks:

            try:

                embedding_response = embedding_client.embeddings.create(
                    model=AZURE_EMBEDDING_DEPLOYMENT,
                    input=chunk["content"],
                )

                vector = embedding_response.data[0].embedding

                search_documents.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "document_id": chunk["document_id"],
                        "source_file": chunk["source_file"],
                        "content": chunk["content"],
                        "content_vector": vector,
                        "document_type": chunk["document_type"],
                        "category": chunk["category"],
                        "section": "",
                        "chunk_index": chunk["chunk_index"],
                    }
                )

            except Exception as e:

                logger.exception(
                    "Embedding failed | " "document_id=%s | chunk_id=%s",
                    document_id,
                    chunk["chunk_id"],
                )

                failed_chunks.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "error": str(e),
                    }
                )

        if not search_documents:

            raise HTTPException(
                status_code=502,
                detail="Embedding generation failed for all chunks.",
            )

        logger.info(
            "Embeddings generated | " "document_id=%s | count=%d",
            document_id,
            len(search_documents),
        )

        # Upsert into AI search service

        try:

            results = search_client.merge_or_upload_documents(
                documents=search_documents
            )

        except AzureError:

            logger.exception(
                "Azure AI Search indexing failed | " "document_id=%s",
                document_id,
            )

            raise HTTPException(
                status_code=502,
                detail="Azure AI Search indexing failed.",
            )

        # check for any indexing failures

        indexing_failures = []

        for result in results:

            if not result.succeeded:

                indexing_failures.append(
                    {
                        "chunk_id": result.key,
                        "error": result.error_message,
                    }
                )

        indexed_count = len(search_documents) - len(indexing_failures)

        logger.info(
            "Indexing completed | " "document_id=%s | indexed=%d | failed=%d",
            document_id,
            indexed_count,
            len(indexing_failures),
        )

        # return response
        return {
            "status": (
                "completed"
                if not failed_chunks and not indexing_failures
                else "completed_with_errors"
            ),
            "document_id": document_id,
            "search_index": AZURE_SEARCH_INDEX_NAME,
            "embedding_model": AZURE_EMBEDDING_DEPLOYMENT,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "chunks_created": len(chunks),
            "chunks_embedded": len(search_documents),
            "chunks_indexed": indexed_count,
            "embedding_failures": failed_chunks,
            "indexing_failures": indexing_failures,
        }

    except HTTPException:
        raise

    except Exception:

        logger.exception(
            "Unexpected embedding error | " "document_id=%s",
            document_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Document embedding failed.",
        )
