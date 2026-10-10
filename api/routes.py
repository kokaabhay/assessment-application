# Necessary imports
from fastapi import FastAPI, HTTPException, APIRouter, UploadFile, File
from api.response_object import Response_Object
from api.doc_extractors import *
from pathlib import Path
import uuid
import os
from typing import Optional, List
from pydantic import Field, BaseModel
from app.llm.query_rewriter import get_rewritten_query
from app.llm.prompt import build_prompt
from app.llm.service import get_answer
from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.reranker import rerank_documents
from app.llm.prompt import build_context
from app.retrieval.hybrid_search import hyde_retrieval
from app.llm.hyde import get_hypothetical_answer
from app.agents.decide_retrieval import get_retrieval_decision
from app.agents.orchestrator import Orchestrator
from azure.storage.blob import BlobServiceClient
from azure.search.documents.indexes import SearchIndexerClient
from azure.core.credentials import AzureKeyCredential
import logging

# Azure Exceptions
from azure.core.exceptions import (
    AzureError,
    ResourceNotFoundError,
)

logger = logging.getLogger(__name__)
from config import *

router = APIRouter(tags=["API"])


@router.post("/get_response")
def response(response_object: Response_Object):
    query = response_object.query
    if not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )
    k = get_retrieval_decision(query)
    # print(k)
    if k == "None":
        return "This content is not permissible for processing by our regulations"
    if k == "True":
        orchestrator = Orchestrator()
        rewritten_query = get_rewritten_query(query)
        decision = orchestrator.get_decision(query, rewritten_query)

        logger.info("\nOriginal query:")
        logger.info(query)

        logger.info("\nRewritten query:")
        logger.info(rewritten_query)

        hypothetical_answer = get_hypothetical_answer(rewritten_query, query)
        logger.info("\nHypothetical answer:")
        logger.info(hypothetical_answer)

        documents = hybrid_search(query, rewritten_query)
        logger.info(f"\nRetrieved {len(documents)} documents.")

        h_documents = hyde_retrieval(hypothetical_answer) if hypothetical_answer else []

        # for i in documents:
        #     print(i)
        reranked_documents = rerank_documents(
            query=query,
            documents=documents,
            top_k=5,
        )

        context_documents = build_context(reranked_documents)
        # print("context documents: ",context_documents)
        logger.info("\nReranked documents:\n")

        for i, document in enumerate(
            reranked_documents,
            start=1,
        ):
            logger.info(f"--- Result {i} ---")
            logger.info(f"Source: {document['source']}")
            logger.info(f"Search score: {document['score']}")
            if document["rerank_score"]:
                logger.info(f"Rerank score: {document['rerank_score']}")
            logger.info(document["content"][:500])
            logger.info("")

        reranked_h_documents = rerank_documents(
            query=query,
            documents=h_documents,
            top_k=5,
        )

        # context_h_documents=build_context(reranked_h_documents)

        prompt = build_prompt(
            query=query,
            documents=context_documents,
            h_documents=reranked_h_documents,
        )

        answer = get_answer(prompt, decision["deployment"])
        logger.info(decision["deployment"], decision["reason"])
        logger.info("\n" + "=" * 60)
        logger.info("FINAL ANSWER")
        logger.info("=" * 60)
        logger.info(answer)

        return (
            "Re-written query is : "
            + rewritten_query
            + "\n\nHypothetical answer:"
            + hypothetical_answer
            + "\n\nReason: "
            + decision["deployment"]
            + decision["reason"]
            + " \n\n LLM response is :"
            + answer
        )
    else:
        context_documents = []
        reranked_h_documents = []
        prompt = build_prompt(
            query=query,
            documents=context_documents,
            h_documents=reranked_h_documents,
        )

        answer = get_answer(prompt, decision["deployment"] if k else "gpt-4.1")
        return " \n\n LLM response is :" + answer


# Connect to Blob Storage
blob_service_client = BlobServiceClient.from_connection_string(
    AZURE_STORAGE_CONNECTION_STRING
)


@router.post("/documents/upload")
async def upload_document(files: List[UploadFile] = File(...)):
    allowed_extensions = {
        ".pdf",
        ".docx",
        ".txt",
        ".md",
        ".xlsx",
        ".csv",
    }

    if not AZURE_STORAGE_CONTAINER3:
        raise HTTPException(
            status_code=500,
            detail="Azure Storage container is not configured.",
        )

    uploaded_files = []

    for file in files:

        original_filename = file.filename

        if not original_filename:
            raise HTTPException(
                status_code=400,
                detail="Filename is required.",
            )

        if "." not in original_filename:
            raise HTTPException(
                status_code=400,
                detail=f"File '{original_filename}' has no extension.",
            )

        extension = "." + original_filename.rsplit(".", 1)[-1].lower()

        if extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=(f"File '{original_filename}' is not supported."),
            )

        # Generate a unique document ID
        document_id = str(uuid.uuid4())

        # Unique blob name
        blob_name = f"{document_id}{extension}"

        try:
            logger.info(
                "Uploading document '%s' with ID '%s'",
                original_filename,
                document_id,
            )

            blob_client = blob_service_client.get_blob_client(
                container=AZURE_STORAGE_CONTAINER3,
                blob=blob_name,
            )

            file_content = await file.read()

            if not file_content:
                raise HTTPException(
                    status_code=400,
                    detail=f"File '{original_filename}' is empty.",
                )

            blob_client.upload_blob(
                file_content,
                overwrite=False,
            )

            logger.info(
                "Blob uploaded successfully: %s",
                blob_client.url,
            )
            # search_indexer_client.run_indexer(AZURE_SEARCH_INDEXER1)
            uploaded_files.append(
                {
                    "document_id": document_id,
                    "blob_name": blob_name,
                    "filename": original_filename,
                    "url": blob_client.url,
                }
            )

        except HTTPException:
            raise

        except Exception as e:
            logger.exception(
                "Failed to upload '%s'",
                original_filename,
            )

            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload '{original_filename}'.",
            )

    return {
        "message": "Documents uploaded successfully.",
        "count": len(uploaded_files),
        "files": uploaded_files,
    }


@router.post("/documents/process/{document_id}")
async def process_document(document_id: str):
    """
    Retrieve a document from Blob Container 3,
    extract its text, and store the extracted
    text as a .txt file in Blob Container 4.
    """

    logger.info(
        "Document processing request received | " "document_id=%s",
        document_id,
    )

    if not document_id:
        raise HTTPException(
            status_code=400,
            detail="Document identifier is required.",
        )

    try:

        # Get the source container
        source_container_client = blob_service_client.get_container_client(
            AZURE_STORAGE_CONTAINER3
        )

        # Get the output container
        output_container_client = blob_service_client.get_container_client(
            AZURE_STORAGE_CONTAINER4
        )

        # Find the source blob
        source_blob_name = None

        logger.info(
            "Searching source blob | " "document_id=%s",
            document_id,
        )

        blobs = source_container_client.list_blobs(name_starts_with=document_id)

        for blob in blobs:
            source_blob_name = blob.name
            break

        if not source_blob_name:
            logger.warning(
                "Source blob not found | " "document_id=%s",
                document_id,
            )

            raise HTTPException(
                status_code=404,
                detail="Document not found.",
            )

        logger.info(
            "Source blob found | " "document_id=%s | blob=%s",
            document_id,
            source_blob_name,
        )

        # Download the source blob

        source_blob_client = source_container_client.get_blob_client(source_blob_name)

        try:
            file_content = source_blob_client.download_blob().readall()

        except ResourceNotFoundError:
            logger.warning(
                "Source blob disappeared during " "processing | document_id=%s",
                document_id,
            )

            raise HTTPException(
                status_code=404,
                detail="Source document not found.",
            )

        except AzureError:
            logger.exception(
                "Azure Blob Storage retrieval failed | " "document_id=%s",
                document_id,
            )

            raise HTTPException(
                status_code=502,
                detail="Unable to retrieve document from storage.",
            )

        # Validate if there is empty document

        if not file_content:
            logger.warning(
                "Empty document | " "document_id=%s",
                document_id,
            )
            raise HTTPException(
                status_code=422,
                detail="The uploaded document is empty.",
            )

        # Detect the file type
        original_filename = Path(source_blob_name).name
        try:
            file_type = detect_file_type(
                file_content,
                original_filename,
            )
        except ValueError as e:
            logger.warning(
                "Unsupported document type | " "document_id=%s | reason=%s",
                document_id,
                str(e),
            )
            raise HTTPException(
                status_code=415,
                detail="Unsupported or invalid document format.",
            )
        logger.info(
            "Document type detected | " "document_id=%s | type=%s",
            document_id,
            file_type,
        )

        # Extract the text
        logger.info(
            "Text extraction started | " "document_id=%s | type=%s",
            document_id,
            file_type,
        )

        try:

            extracted_text = extract_text(
                file_content,
                file_type,
            )

        except ValueError as e:

            logger.warning(
                "Document extraction failed | " "document_id=%s | reason=%s",
                document_id,
                str(e),
            )

            raise HTTPException(
                status_code=422,
                detail=(
                    "The document is malformed, " "corrupted, or could not be read."
                ),
            )

        except Exception:

            logger.exception(
                "Unexpected extraction failure | " "document_id=%s",
                document_id,
            )

            raise HTTPException(
                status_code=500,
                detail="Document text extraction failed.",
            )

        # Validation of the extracted text

        if not extracted_text.strip():
            logger.warning(
                "No meaningful text extracted | " "document_id=%s",
                document_id,
            )
            raise HTTPException(
                status_code=422,
                detail=(
                    "The document was read successfully, "
                    "but no meaningful text could be extracted."
                ),
            )

        logger.info(
            "Text extraction completed | " "document_id=%s | characters=%d",
            document_id,
            len(extracted_text),
        )

        # create the output blob name
        output_blob_name = f"{document_id}.txt"

        output_blob_client = output_container_client.get_blob_client(output_blob_name)

        # store Extracted text
        logger.info(
            "Saving extracted text | " "document_id=%s | output_blob=%s",
            document_id,
            output_blob_name,
        )

        try:
            output_blob_client.upload_blob(
                extracted_text.encode("utf-8"),
                overwrite=True,
            )

        except AzureError:
            logger.exception(
                "Failed to store extracted text | " "document_id=%s",
                document_id,
            )

            raise HTTPException(
                status_code=502,
                detail=("Failed to store extracted text " "in Blob Storage."),
            )

        # log completion
        logger.info(
            "Document processing completed successfully | "
            "document_id=%s | output_blob=%s",
            document_id,
            output_blob_name,
        )

        return {
            "status": "completed",
            "document_id": document_id,
            "source_blob": source_blob_name,
            "output_container": AZURE_STORAGE_CONTAINER4,
            "extracted_text_blob": output_blob_name,
            "message": (
                "Document processed successfully " "and extracted text was stored."
            ),
        }

    except HTTPException:
        raise

    except AzureError:

        logger.exception(
            "Azure Storage error | " "document_id=%s",
            document_id,
        )

        raise HTTPException(
            status_code=502,
            detail="Azure Blob Storage operation failed.",
        )

    except Exception:

        logger.exception(
            "Unexpected document processing error | " "document_id=%s",
            document_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Document processing failed.",
        )
