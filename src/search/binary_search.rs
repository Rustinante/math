use std::{cmp::Ordering, ops::Index};
use Ordering::{Equal, Greater, Less};

pub trait BinarySearch: Index<usize> {
    /// performs binary search between the `start` and `end` indices
    /// `start`: start index
    /// `end`: end index exclusive
    /// `cmp`: a function that returns `Ordering::Less` when the first argument is less than the
    /// second, etc. returns the index of the target element as `Some(usize)` if present, `Err`
    /// otherwise. In the case of returning an `Err`, the associated value is `Some(usize)`
    /// representing the index at which the target value can be inserted while maintaining the
    /// sorted order, or `None` if the provided `[start, end)` range is empty, i.e. when `start
    /// >= end`.
    fn binary_search_with_cmp<E, F>(
        &self,
        start: usize,
        end: usize,
        target: &E,
        cmp: F,
    ) -> Result<usize, Option<usize>>
    where
        F: Fn(&<Self as Index<usize>>::Output, &E) -> Ordering;
}

impl<T> BinarySearch for Vec<T> {
    fn binary_search_with_cmp<E, F>(
        &self,
        start: usize,
        end_exclusive: usize,
        target: &E,
        cmp: F,
    ) -> Result<usize, Option<usize>>
    where
        F: Fn(&<Self as Index<usize>>::Output, &E) -> Ordering, {
        if start >= end_exclusive {
            return Err(None);
        }
        let mut start = start;
        // now end is inclusive
        let mut end = end_exclusive - 1;
        if cmp(&self[start], target) == Ordering::Greater {
            return Err(Some(start));
        }
        if cmp(&self[end], target) == Ordering::Less {
            return Err(Some(end + 1));
        }
        let mut mid = (start + end) / 2;
        while end > start {
            match cmp(&self[mid], target) {
                Less => {
                    start = mid + 1;
                    mid = (start + end) / 2;
                }
                Greater => {
                    end = mid;
                    mid = (start + end) / 2;
                }
                Equal => return Ok(mid),
            };
        }
        match cmp(&self[mid], target) {
            Equal => Ok(mid),
            Less => Err(Some(mid + 1)),
            Greater => Err(Some(mid)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BinarySearch;

    #[test]
    fn test_vec_binary_search() {
        let v = vec![1, 2, 3, 6, 29, 43, 69, 100, 340];
        assert_eq!(
            BinarySearch::binary_search_with_cmp(&v, 0, v.len(), &30, |x, y| x.cmp(y)),
            Err(Some(5))
        );
        assert_eq!(
            BinarySearch::binary_search_with_cmp(&v, 0, v.len(), &-10, |x, y| x.cmp(y)),
            Err(Some(0))
        );
        assert_eq!(
            BinarySearch::binary_search_with_cmp(&v, 0, v.len(), &0, |x, y| x.cmp(y)),
            Err(Some(0))
        );
        assert_eq!(
            BinarySearch::binary_search_with_cmp(&v, 0, v.len(), &341, |x, y| x.cmp(y)),
            Err(Some(v.len()))
        );
        assert_eq!(
            BinarySearch::binary_search_with_cmp(&v, 0, v.len(), &1000, |x, y| x.cmp(y)),
            Err(Some(v.len()))
        );
        assert_eq!(
            BinarySearch::binary_search_with_cmp(&v, 0, v.len(), &1, |x, y| x.cmp(y)),
            Ok(0)
        );
        assert_eq!(
            BinarySearch::binary_search_with_cmp(&v, 0, v.len(), &2, |x, y| x.cmp(y)),
            Ok(1)
        );
        assert_eq!(
            BinarySearch::binary_search_with_cmp(&v, 0, v.len(), &3, |x, y| x.cmp(y)),
            Ok(2)
        );
        assert_eq!(
            BinarySearch::binary_search_with_cmp(&v, 0, v.len(), &6, |x, y| x.cmp(y)),
            Ok(3)
        );
        assert_eq!(
            BinarySearch::binary_search_with_cmp(&v, 0, v.len(), &29, |x, y| x.cmp(y)),
            Ok(4)
        );
        assert_eq!(
            BinarySearch::binary_search_with_cmp(&v, 0, v.len(), &43, |x, y| x.cmp(y)),
            Ok(5)
        );
        assert_eq!(
            BinarySearch::binary_search_with_cmp(&v, 0, v.len(), &69, |x, y| x.cmp(y)),
            Ok(6)
        );
        assert_eq!(
            BinarySearch::binary_search_with_cmp(&v, 0, v.len(), &100, |x, y| x.cmp(y)),
            Ok(7)
        );
        assert_eq!(
            BinarySearch::binary_search_with_cmp(&v, 0, v.len(), &340, |x, y| x.cmp(y)),
            Ok(8)
        );
        assert_eq!(
            BinarySearch::binary_search_with_cmp(&v, 3, 6, &6, |x, y| x.cmp(y)),
            Ok(3)
        );
        assert_eq!(
            BinarySearch::binary_search_with_cmp(&v, 3, 6, &29, |x, y| x.cmp(y)),
            Ok(4)
        );
        assert_eq!(
            BinarySearch::binary_search_with_cmp(&v, 3, 6, &43, |x, y| x.cmp(y)),
            Ok(5)
        );
        assert_eq!(
            BinarySearch::binary_search_with_cmp(&v, 3, 6, &340, |x, y| x.cmp(y)),
            Err(Some(6))
        );
        assert_eq!(
            BinarySearch::binary_search_with_cmp(&v, 3, 6, &2, |x, y| x.cmp(y)),
            Err(Some(3))
        );
        assert_eq!(
            BinarySearch::binary_search_with_cmp(&v, 3, 6, &3, |x, y| x.cmp(y)),
            Err(Some(3))
        );
        assert_eq!(
            BinarySearch::binary_search_with_cmp(&v, 3, 6, &69, |x, y| x.cmp(y)),
            Err(Some(6))
        );
        let v = vec![-10];
        assert_eq!(
            BinarySearch::binary_search_with_cmp(&v, 0, v.len(), &9, |x, y| x.cmp(y)),
            Err(Some(1))
        );
        assert_eq!(
            BinarySearch::binary_search_with_cmp(&v, 0, v.len(), &-10, |x, y| x.cmp(y)),
            Ok(0)
        );
        let v = vec![-10, 1];
        assert_eq!(
            BinarySearch::binary_search_with_cmp(&v, 0, v.len(), &1, |x, y| x.cmp(y)),
            Ok(1)
        );
        assert_eq!(
            BinarySearch::binary_search_with_cmp(&v, 0, v.len(), &-10, |x, y| x.cmp(y)),
            Ok(0)
        );
    }
}
