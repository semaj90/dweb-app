//go:build legacy
// +build legacy

package minio

import (
	"context"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"strings"
	"time"

	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
)

type Client struct {
	client     *minio.Client
	bucketName string
}

type UploadResult struct {
	ObjectName string            `json:"objectName"`
	Size       int64             `json:"size"`
	ETag       string            `json:"etag"`
	URL        string            `json:"url"`
	Metadata   map[string]string `json:"metadata"`
}

type UploadOptions struct {
	CaseID       string            `json:"caseId"`
	DocumentType string            `json:"documentType"`
	Tags         map[string]string `json:"tags"`
	Metadata     map[string]string `json:"metadata"`
}

// NewClient creates a new MinIO client
func NewClient(endpoint, accessKey, secretKey, bucketName string, useSSL bool) (*Client, error) {
	minioClient, err := minio.New(endpoint, &minio.Options{
		Creds:  credentials.NewStaticV4(accessKey, secretKey, ""),
		Secure: useSSL,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create MinIO client: %w", err)
	}

	client := &Client{
		client:     minioClient,
		bucketName: bucketName,
	}

	// Ensure bucket exists
	if err := client.ensureBucket(); err != nil {
		return nil, fmt.Errorf("failed to ensure bucket exists: %w", err)
	}

	return client, nil
}

// ensureBucket creates the bucket if it doesn't exist
func (c *Client) ensureBucket() error {
	ctx := context.Background()
	exists, err := c.client.BucketExists(ctx, c.bucketName)
	if err != nil {
		return err
	}

	if !exists {
		err = c.client.MakeBucket(ctx, c.bucketName, minio.MakeBucketOptions{})
		if err != nil {
			return err
		}
		log.Printf("Created bucket: %s", c.bucketName)
	}

	return nil
}

// UploadFile uploads a file with metadata for legal document processing
func (c *Client) UploadFile(ctx context.Context, file multipart.File, header *multipart.FileHeader, opts UploadOptions) (*UploadResult, error) {
	// Generate unique object name with case organization
	timestamp := time.Now().Format("2006/01/02/15-04-05")
	objectName := fmt.Sprintf("cases/%s/%s/%s-%s",
		opts.CaseID,
		opts.DocumentType,
		timestamp,
		header.Filename)

	// Prepare metadata (sanitize to avoid reserved header names)
	metadata := map[string]string{
		"case-id":       opts.CaseID,
		"document-type": opts.DocumentType,
		"filename":      header.Filename,
		"upload-time":   time.Now().Format(time.RFC3339),
		"file-size":     fmt.Sprintf("%d", header.Size),
	}

	// Add custom metadata
	// Filter out reserved or disallowed keys like "content-type", "content-length", etc.
	if opts.Metadata != nil {
		for k, v := range opts.Metadata {
			lk := strings.ToLower(k)
			// Disallow common reserved headers and any x-amz-* which S3 uses internally
			if lk == "content-type" || lk == "content-length" || lk == "etag" || lk == "last-modified" ||
				lk == "cache-control" || lk == "content-encoding" || lk == "content-language" ||
				lk == "content-disposition" || lk == "expires" || strings.HasPrefix(lk, "x-amz-") {
				continue
			}
			metadata[k] = v
		}
	}

	// Add tags
	tags := make(map[string]string)
	tags["case"] = opts.CaseID
	tags["type"] = opts.DocumentType
	for k, v := range opts.Tags {
		tags[k] = v
	}

	// Upload options
	uploadOpts := minio.PutObjectOptions{
		UserMetadata: metadata,
		UserTags:     tags,
		ContentType:  header.Header.Get("Content-Type"),
	}

	// Reset file reader to beginning
	file.Seek(0, 0)

	// Upload the file
	info, err := c.client.PutObject(ctx, c.bucketName, objectName, file, header.Size, uploadOpts)
	if err != nil {
		return nil, fmt.Errorf("failed to upload file: %w", err)
	}

	// Generate presigned URL for access (valid for 24 hours)
	url, err := c.client.PresignedGetObject(ctx, c.bucketName, objectName, 24*time.Hour, nil)
	if err != nil {
		log.Printf("Warning: failed to generate presigned URL: %v", err)
		url = nil
	}

	// Include content-type in returned metadata for DB persistence (not sent as user metadata)
	ct := header.Header.Get("Content-Type")
	if ct != "" {
		metadata["content-type"] = ct
	}

	result := &UploadResult{
		ObjectName: objectName,
		Size:       info.Size,
		ETag:       info.ETag,
		URL:        "",
		Metadata:   metadata,
	}

	if url != nil {
		result.URL = url.String()
	}

	return result, nil
}

// GetFile retrieves a file from MinIO
func (c *Client) GetFile(ctx context.Context, objectName string) (io.ReadCloser, *minio.ObjectInfo, error) {
	object, err := c.client.GetObject(ctx, c.bucketName, objectName, minio.GetObjectOptions{})
	if err != nil {
		return nil, nil, fmt.Errorf("failed to get object: %w", err)
	}

	info, err := object.Stat()
	if err != nil {
		object.Close()
		return nil, nil, fmt.Errorf("failed to get object info: %w", err)
	}

	return object, &info, nil
}

// DeleteFile removes a file from MinIO
func (c *Client) DeleteFile(ctx context.Context, objectName string) error {
	return c.client.RemoveObject(ctx, c.bucketName, objectName, minio.RemoveObjectOptions{})
}

// ListFiles lists files for a specific case
func (c *Client) ListFiles(ctx context.Context, caseID string) ([]minio.ObjectInfo, error) {
	prefix := fmt.Sprintf("cases/%s/", caseID)

	var objects []minio.ObjectInfo
	for object := range c.client.ListObjects(ctx, c.bucketName, minio.ListObjectsOptions{
		Prefix:    prefix,
		Recursive: true,
	}) {
		if object.Err != nil {
			return nil, object.Err
		}
		objects = append(objects, object)
	}

	return objects, nil
}

// GetPresignedUploadURL generates a presigned URL for direct browser upload
func (c *Client) GetPresignedUploadURL(ctx context.Context, objectName string, expires time.Duration) (string, error) {
	url, err := c.client.PresignedPutObject(ctx, c.bucketName, objectName, expires)
	if err != nil {
		return "", fmt.Errorf("failed to generate presigned upload URL: %w", err)
	}
	return url.String(), nil
}

// GetObjectMetadata retrieves metadata for an object
func (c *Client) GetObjectMetadata(ctx context.Context, objectName string) (map[string]string, error) {
	info, err := c.client.StatObject(ctx, c.bucketName, objectName, minio.StatObjectOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to get object metadata: %w", err)
	}
	return info.UserMetadata, nil
}
