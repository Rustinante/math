pub struct ConcatenatedIter<I: Iterator> {
    iters: Vec<I>,
    current_iter_index: usize,
}

impl<I: Iterator> ConcatenatedIter<I> {
    fn from_iters(iters: Vec<I>) -> Self {
        ConcatenatedIter {
            iters,
            current_iter_index: 0,
        }
    }

    pub fn concat_iter(self, other: I) -> Self {
        let mut iters = self.iters;
        iters.push(other);
        ConcatenatedIter {
            iters,
            current_iter_index: self.current_iter_index,
        }
    }
}

impl<I: Iterator> From<Vec<I>> for ConcatenatedIter<I> {
    fn from(iters: Vec<I>) -> Self {
        Self::from_iters(iters)
    }
}

impl<I: Iterator> Iterator for ConcatenatedIter<I> {
    type Item = <I as Iterator>::Item;

    fn next(&mut self) -> Option<Self::Item> {
        match self.iters.get_mut(self.current_iter_index) {
            None => None,
            Some(iter) => match iter.next() {
                Some(x) => Some(x),
                None => {
                    self.current_iter_index += 1;
                    self.next()
                }
            },
        }
    }
}

pub trait IntoConcatIter: Iterator + Sized {
    fn into_concat_iter(self, other: Self) -> ConcatenatedIter<Self> {
        ConcatenatedIter {
            iters: vec![self, other],
            current_iter_index: 0,
        }
    }
}

impl<I: Iterator + Sized> IntoConcatIter for I {}

#[cfg(test)]
mod tests {
    use crate::iter::{concatenated_iter::IntoConcatIter, ConcatenatedIter};
    use std::collections::BTreeMap;

    #[test]
    fn test_from_iters() {
        let arr1 = vec![0, 1, 2];
        let arr2 = vec![3, 4];
        let arr3 = vec![5, 6];
        let mut concat_iter = ConcatenatedIter::from(vec![arr1.iter(), arr2.iter(), arr3.iter()]);
        assert_eq!(concat_iter.next(), Some(&0));
        assert_eq!(concat_iter.next(), Some(&1));
        assert_eq!(concat_iter.next(), Some(&2));
        assert_eq!(concat_iter.next(), Some(&3));
        assert_eq!(concat_iter.next(), Some(&4));
        assert_eq!(concat_iter.next(), Some(&5));
        assert_eq!(concat_iter.next(), Some(&6));
        assert_eq!(concat_iter.next(), None);
    }

    #[test]
    fn test_concat_vec_iter() {
        let arr1 = vec![0, 1, 2];
        let arr2 = vec![3, 4];
        let arr3 = vec![6, 7, 8];
        {
            let mut concat_1_2 = arr1.iter().into_concat_iter(arr2.iter());
            assert_eq!(concat_1_2.next(), Some(&0));
            assert_eq!(concat_1_2.next(), Some(&1));
            assert_eq!(concat_1_2.next(), Some(&2));
            assert_eq!(concat_1_2.next(), Some(&3));
            assert_eq!(concat_1_2.next(), Some(&4));
            assert_eq!(concat_1_2.next(), None);
        }
        {
            let mut concat_1_2 = arr1.iter().into_concat_iter(arr2.iter());
            assert_eq!(concat_1_2.next(), Some(&0));
            assert_eq!(concat_1_2.next(), Some(&1));
            concat_1_2 = concat_1_2.concat_iter(arr3.iter());
            assert_eq!(concat_1_2.next(), Some(&2));
            assert_eq!(concat_1_2.next(), Some(&3));
            assert_eq!(concat_1_2.next(), Some(&4));
            assert_eq!(concat_1_2.next(), Some(&6));
            assert_eq!(concat_1_2.next(), Some(&7));
            concat_1_2 = concat_1_2.concat_iter(arr3.iter());
            assert_eq!(concat_1_2.next(), Some(&8));
            assert_eq!(concat_1_2.next(), Some(&6));
            assert_eq!(concat_1_2.next(), Some(&7));
            assert_eq!(concat_1_2.next(), Some(&8));
            concat_1_2 = concat_1_2.concat_iter(arr1.iter());
            assert_eq!(concat_1_2.next(), Some(&0));
            assert_eq!(concat_1_2.next(), Some(&1));
            assert_eq!(concat_1_2.next(), Some(&2));
            assert_eq!(concat_1_2.next(), None);
        }
    }

    #[test]
    fn test_concat_btreemap_iter() {
        let m1: BTreeMap<i32, i32> = vec![(1, 2), (3, 4)].into_iter().collect();
        let m2: BTreeMap<i32, i32> = vec![(11, 12), (13, 14), (15, 16)].into_iter().collect();
        let mut iter = m1.iter().into_concat_iter(m2.iter());
        assert_eq!(iter.next(), Some((&1, &2)));
        assert_eq!(iter.next(), Some((&3, &4)));
        assert_eq!(iter.next(), Some((&11, &12)));
        assert_eq!(iter.next(), Some((&13, &14)));
        assert_eq!(iter.next(), Some((&15, &16)));
    }
}
